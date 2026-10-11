"""视频的标题、作者、链接，和由此得到的曲名、原唱：曲名、原唱写进谱里，视频作者和链接印在每页页脚（注明出处）。

- B 站：视频信息接口给标题、UP 主（用网页版同样的 WBI 签名接口：没登录时不签名的接口会被拒，HTTP 412）；
  B 站自己识别出的背景音乐（播放器接口的 bgm_info）给曲名和原唱。
- 其他网站：yt-dlp 给标题、上传者，音乐类视频还有曲名（track）和演唱者（artist）。
- 没识别出背景音乐时，从视频标题里取曲名：见 song_from_title。
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass
from typing import Any

from video2score.fetch import _api_headers, bvid_from_source

_VIEW = "https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
_WBI_VIEW = "https://api.bilibili.com/x/web-interface/wbi/view?{query}"
_NAV = "https://api.bilibili.com/x/web-interface/nav"
# WBI 签名：把 nav 接口给的两段密钥按这个顺序重排，取前 32 位（和 B 站网页版的做法一样）。
_MIXIN = (
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
)
_PLAYER = "https://api.bilibili.com/x/player/v2?bvid={bvid}&cid={cid}"
_BGM = "https://api.bilibili.com/x/copyright-music-publicity/bgm/detail?music_id={music_id}"
_TAGS = "https://api.bilibili.com/x/tag/archive/tags?bvid={bvid}"


@dataclass
class VideoInfo:
    link: str
    title: str = ""  # 视频标题
    author: str = ""  # 视频作者（UP 主、频道）
    song: str = ""  # 曲名
    artist: str = ""  # 原唱（演唱者、乐队）

    def credit(self) -> str:
        """每页页脚的出处："视频：作者 链接"。"""
        return "视频：" + "  ".join(item for item in (self.author, self.link) if item)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoInfo:
        return cls(**{key: str(data.get(key) or "") for key in ("link", "title", "author", "song", "artist")})


def lookup(kind: str, ref: str) -> VideoInfo | None:
    """按视频来源查信息；本地文件没有可查的，返回 None。"""
    if kind == "bilibili":
        from video2score.source import _expand_short_link

        return bilibili_info(_expand_short_link(ref))
    if kind == "url":
        return ytdlp_info(ref)
    return None


def bilibili_info(source: str) -> VideoInfo:
    from curl_cffi import requests

    bvid = bvid_from_source(source)
    session = requests.Session(impersonate="chrome124")
    session.get("https://www.bilibili.com/", timeout=20)
    headers = _api_headers(bvid)
    try:
        view = _data(session.get(_WBI_VIEW.format(query=_wbi_sign(session, headers, {"bvid": bvid})), headers=headers, timeout=20), "视频信息")
    except Exception:
        view = _data(session.get(_VIEW.format(bvid=bvid), headers=headers, timeout=20), "视频信息")
    info = VideoInfo(
        link=f"https://www.bilibili.com/video/{bvid}/",
        title=str(view.get("title") or ""),
        author=str((view.get("owner") or {}).get("name") or ""),
    )
    # 曲名、原唱查不到不要紧（下面从标题里取），视频作者和链接有了就行。
    try:
        player = _data(session.get(_PLAYER.format(bvid=bvid, cid=view.get("cid")), headers=headers, timeout=20), "播放器信息")
        bgm = player.get("bgm_info") or {}
        if bgm.get("music_id"):
            detail = _data(session.get(_BGM.format(music_id=bgm["music_id"]), headers=headers, timeout=20), "背景音乐")
            info.song = str(detail.get("music_title") or "")
            info.artist = ", ".join(part.strip() for part in str(detail.get("origin_artist") or "").split(",") if part.strip())
    except Exception:
        pass
    if not info.song:
        try:
            tags = [str(item.get("tag_name") or "") for item in _list(session.get(_TAGS.format(bvid=bvid), headers=headers, timeout=20))]
        except Exception:
            tags = []
        info.song, info.artist = song_from_title(info.title, tags)
    return info


def _wbi_sign(session: Any, headers: dict[str, str], params: dict[str, Any]) -> str:
    """给参数加上 wts（时间戳）和 w_rid（签名），返回查询串。"""
    images = (_body(session.get(_NAV, headers=headers, timeout=20), "nav", allow_error=True) or {}).get("wbi_img") or {}
    raw = "".join(str(images.get(name) or "").rsplit("/", 1)[-1].split(".")[0] for name in ("img_url", "sub_url"))
    if len(raw) < 64:
        raise RuntimeError("没拿到 WBI 密钥")
    mixin = "".join(raw[index] for index in _MIXIN)[:32]
    signed = dict(sorted({**params, "wts": int(time.time())}.items()))
    query = urllib.parse.urlencode(signed)
    return query + "&w_rid=" + hashlib.md5((query + mixin).encode("utf-8")).hexdigest()


def ytdlp_info(url: str) -> VideoInfo:
    import yt_dlp

    from video2score.source import ytdlp_options

    with yt_dlp.YoutubeDL(ytdlp_options()) as downloader:
        data = downloader.extract_info(url, download=False)
    info = VideoInfo(
        link=str(data.get("webpage_url") or url),
        title=str(data.get("title") or ""),
        author=str(data.get("uploader") or data.get("channel") or ""),
        song=str(data.get("track") or ""),
        artist=str(data.get("artist") or data.get("creator") or ""),
    )
    if not info.song:
        info.song, info.artist = song_from_title(info.title, [str(tag) for tag in data.get("tags") or []])
    return info


def song_from_title(title: str, tags: list[str], online: bool = True) -> tuple[str, str]:
    """从视频标题里取 (曲名, 原唱)。

    去掉【】、[] 里的栏目字样和表情符号后：有《》就取书名号里的曲名；按 " / " 分段时，相邻两段"曲名 / 原唱"得到证实才用：
    第二段是视频标签之一（乐队、歌手名常打成标签），或者 MusicBrainz（开放的音乐资料库）里有这个原唱的这首歌（online 时查）。
    都不是就把整个标题当曲名，原唱空着（宁可不填，不猜错）。
    """
    text = re.sub(r"【[^】]*】|\[[^\]]*\]", " ", title)
    # 表情符号（💕 之类）谱上的字体大多画不出，去掉。
    text = " ".join("".join(" " if unicodedata.category(char) in ("So", "Cs") or 0xFE00 <= ord(char) <= 0xFE0F else char for char in text).split())
    named = re.search(r"《([^》]+)》", text)
    if named:
        return named.group(1).strip(), ""
    parts = [part.strip() for part in re.split(r"\s+/\s+", text) if part.strip()]
    known = {_plain(tag) for tag in tags if tag.strip()}
    if len(parts) >= 2 and _plain(parts[1]) in known:
        return parts[0], parts[1]
    if online:
        for song, artist in list(zip(parts, parts[1:]))[:3]:
            if _musicbrainz_has(song, artist):
                return song, artist
    return text, ""


_MUSICBRAINZ = "https://musicbrainz.org/ws/2/recording?{query}"


def _musicbrainz_has(song: str, artist: str) -> bool:
    """MusicBrainz 里有没有 artist 唱的 song（名字去掉空格、不分大小写后一样）。查不到、出错都当没有。

    MusicBrainz 要求每秒最多一次请求、带上能联系到的 User-Agent。
    """
    import json
    import urllib.request

    from video2score import __version__

    query = urllib.parse.urlencode({"query": f'recording:"{song}" AND artist:"{artist}"', "fmt": "json", "limit": 5})
    request = urllib.request.Request(
        _MUSICBRAINZ.format(query=query), headers={"User-Agent": f"video2score/{__version__} ( https://github.com/shxiao-archive/video2score )"}
    )
    time.sleep(1.0)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            found = json.loads(response.read().decode("utf-8")).get("recordings") or []
    except Exception:
        return False
    return any(
        _plain(item.get("title") or "") == _plain(song)
        and any(_plain(credit.get("name") or "") == _plain(artist) for credit in item.get("artist-credit") or [])
        for item in found
        if int(item.get("score") or 0) >= 90
    )


def _plain(text: str) -> str:
    return "".join(text.split()).casefold()


def _data(response: Any, label: str) -> dict:
    data = _body(response, label)
    return data if isinstance(data, dict) else {}


def _list(response: Any) -> list:
    data = _body(response, "标签")
    return data if isinstance(data, list) else []


def _body(response: Any, label: str, allow_error: bool = False) -> Any:
    """接口回的 data。allow_error：code 不是 0 也照样取 data（nav 没登录时回 -101，但 data 里有 WBI 密钥）。"""
    if response.status_code != 200:
        raise RuntimeError(f"{label} HTTP {response.status_code}")
    body = response.json()
    if not allow_error and body.get("code") not in (0, None):
        raise RuntimeError(f"{label}：{body.get('message') or body.get('code')}")
    return body.get("data")
