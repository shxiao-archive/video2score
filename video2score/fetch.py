"""把粘贴的哔哩哔哩链接解析成视频文件。

只用分 P 列表和 playurl。不请求会在当前网络返回 412 的 view 接口。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from video2score.paths import COOKIE_PATH

_BVID_RE = re.compile(r"BV[0-9A-Za-z]{10}")
_PAGELIST = "https://api.bilibili.com/x/player/pagelist?bvid={bvid}"
_PLAYURL = (
    "https://api.bilibili.com/x/player/playurl"
    "?bvid={bvid}&cid={cid}&qn={qn}&fnval=0&fnver=0&fourk=1"
)
_PLAYURL_DASH = (
    "https://api.bilibili.com/x/player/playurl"
    "?bvid={bvid}&cid={cid}&qn=116&fnval=4048&fnver=0&fourk=1"
)
_QN_TRY = (116, 112, 80, 64)
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def bvid_from_source(source: str) -> str:
    match = _BVID_RE.search(source or "")
    if not match:
        raise ValueError(f"链接里没有 BV 号：{source}")
    return match.group(0)


def download_bilibili(source: str, dest: Path) -> Path:
    """下载视频到 dest，返回写入的路径。文件必须非空。"""
    from curl_cffi import requests

    bvid = bvid_from_source(source)
    session = requests.Session(impersonate="chrome124")
    session.get("https://www.bilibili.com/", timeout=20)
    headers = _api_headers(bvid)
    page = session.get(_PAGELIST.format(bvid=bvid), headers=headers, timeout=20)
    payload = _json_body(page, "pagelist")
    rows = payload.get("data") or []
    if payload.get("code") not in (0, None) or not rows or not rows[0].get("cid"):
        raise RuntimeError(f"分 P 列表没有 cid：{payload.get('message') or payload}")
    cid = int(rows[0]["cid"])
    cdn_headers = {"Referer": headers["Referer"], "User-Agent": headers["User-Agent"]}
    dash = _select_dash(session, bvid, cid, headers)
    if dash is not None:
        media_url, got_qn = dash
        expected = _remote_size(session, media_url, cdn_headers)
    else:
        stream, got_qn = _select_stream(session, bvid, cid, headers)
        media_url = str(stream["url"])
        expected = int(stream.get("size") or 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            _download_ranges(session, media_url, cdn_headers, dest, expected)
            dest.with_suffix(".qn").write_text(str(got_qn), encoding="utf-8")
            return dest
        except Exception as exc:
            last_error = exc
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"下载视频失败：{last_error}")


def _download_ranges(
    session: Any,
    url: str,
    headers: dict[str, str],
    dest: Path,
    expected: int,
) -> None:
    """按区间下载。整文件 HTTP/2 流在这几个 CDN 上会挂起或被重置。"""
    from curl_cffi.const import CurlHttpVersion

    if expected < 1000:
        raise RuntimeError(f"playurl 没有给出文件大小：{expected}")
    partial = dest.with_suffix(dest.suffix + ".part")
    chunk = 2 * 1024 * 1024
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected or start % chunk != 0:
        partial.unlink(missing_ok=True)
        start = 0
    try:
        with partial.open("ab" if start else "wb") as handle:
            while start < expected:
                end = min(start + chunk, expected) - 1
                data = _read_range(session, url, headers, start, end, CurlHttpVersion.V1_1)
                if start == 0 and not _is_media(data):
                    snippet = data[:200].decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"playurl 媒体不是视频：{' '.join(snippet.split())}"
                    )
                if len(data) != end - start + 1:
                    raise RuntimeError(
                        f"区间 {start}-{end} 长度不符：{len(data)}"
                    )
                handle.write(data)
                start += len(data)
        if partial.stat().st_size < 1000:
            raise RuntimeError(f"下载到的视频过小：{partial.stat().st_size} 字节。")
        partial.replace(dest)
    except Exception:
        if start == 0 and partial.exists():
            partial.unlink()
        raise


def _read_range(
    session: Any,
    url: str,
    headers: dict[str, str],
    start: int,
    end: int,
    http_version: Any,
) -> bytes:
    last_error: Exception | None = None
    ranged = dict(headers)
    ranged["Range"] = f"bytes={start}-{end}"
    for _attempt in range(3):
        try:
            response = session.get(
                url,
                headers=ranged,
                timeout=40,
                http_version=http_version,
            )
            data = response.content or b""
            if response.status_code == 200 and start == 0:
                return data
            if response.status_code != 206 or not data:
                snippet = " ".join((response.text or "")[:200].split())
                raise RuntimeError(
                    f"区间 {start}-{end} HTTP {response.status_code}: {snippet}"
                )
            return data
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"区间 {start}-{end} 下载失败：{last_error}")


def load_cookie() -> str | None:
    if not COOKIE_PATH.is_file():
        return None
    text = COOKIE_PATH.read_text(encoding="utf-8").strip()
    return text or None


def _api_headers(bvid: str) -> dict[str, str]:
    headers = {
        "Referer": f"https://www.bilibili.com/video/{bvid}/",
        "User-Agent": _UA,
    }
    cookie = load_cookie()
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _select_dash(
    session: Any, bvid: str, cid: int, headers: dict[str, str]
) -> tuple[str, int] | None:
    """登录后高清晰度在 DASH 里。只要 AVC，OpenCV 读不了 HEVC。"""
    play = session.get(_PLAYURL_DASH.format(bvid=bvid, cid=cid), headers=headers, timeout=20)
    body = _json_body(play, "playurl")
    videos = ((body.get("data") or {}).get("dash") or {}).get("video") or []
    best_url = ""
    best_qn = -1
    for item in videos:
        codecs = str(item.get("codecs") or "")
        url = item.get("baseUrl") or item.get("base_url") or ""
        qn = int(item.get("id") or 0)
        if not url or not codecs.startswith("avc1"):
            continue
        if qn > best_qn:
            best_qn = qn
            best_url = str(url)
    if not best_url:
        return None
    return best_url, best_qn


def _remote_size(session: Any, url: str, headers: dict[str, str]) -> int:
    from curl_cffi.const import CurlHttpVersion

    ranged = dict(headers)
    ranged["Range"] = "bytes=0-0"
    response = session.get(url, headers=ranged, timeout=20, http_version=CurlHttpVersion.V1_1)
    span = str(response.headers.get("content-range") or "")
    total = span.rsplit("/", 1)[-1] if "/" in span else ""
    if total.isdigit() and int(total) >= 1000:
        return int(total)
    length = str(response.headers.get("content-length") or "")
    if response.status_code == 200 and length.isdigit() and int(length) >= 1000:
        return int(length)
    raise RuntimeError(f"无法确认视频大小：HTTP {response.status_code} {span}")


def _select_stream(
    session: Any, bvid: str, cid: int, headers: dict[str, str]
) -> tuple[dict[str, Any], int]:
    """从高到低要单文件 MP4。接口降级时继续试下一档，留下实际给到的最高档。"""
    best_stream: dict[str, Any] | None = None
    best_qn = -1
    for qn in _QN_TRY:
        play = session.get(
            _PLAYURL.format(bvid=bvid, cid=cid, qn=qn),
            headers=headers,
            timeout=20,
        )
        body = _json_body(play, "playurl")
        data = body.get("data") or {}
        streams = data.get("durl") or []
        if body.get("code") not in (0, None) or not streams or not streams[0].get("url"):
            continue
        got = int(data.get("quality") or 0)
        if got > best_qn:
            best_qn = got
            best_stream = streams[0]
        if got >= qn:
            break
    if best_stream is None:
        raise RuntimeError("playurl 没有视频地址。")
    return best_stream, best_qn


def _json_body(response: Any, label: str) -> dict:
    if response.status_code != 200:
        snippet = " ".join((response.text or "").split())[:300]
        raise RuntimeError(f"{label} HTTP {response.status_code}: {snippet}")
    return response.json()


def _is_media(content: bytes) -> bool:
    if len(content) < 12:
        return False
    if content[4:8] == b"ftyp":
        return True
    return content.startswith((b"FLV", b"RIFF", b"OggS", b"\x1a\x45\xdf\xa3"))
