"""哔哩哔哩扫码登录。Cookie 只写在 COOKIE_PATH（仓库根目录或 video2score/data/config 里的 bilibili_cookie.txt），不打印。"""

from __future__ import annotations

import time
from pathlib import Path

from video2score.fetch import COOKIE_PATH

_GENERATE = (
    "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
    "?source=main-fe-header&go_url=https:%2F%2Fwww.bilibili.com%2F"
    "&web_location=333.1007"
)
_POLL = (
    "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
    "?qrcode_key={key}&source=main_web&web_location=333.1228"
)


def generate_login_qr(dest: Path) -> str:
    """把登录二维码写成 PNG，返回 qrcode_key。"""
    png, key = login_qr()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(png)
    return key


def login_qr() -> tuple[bytes, str]:
    """登录二维码（PNG 数据）和 qrcode_key；不写文件，界面直接显示。"""
    import io

    import qrcode
    from curl_cffi import requests

    session = requests.Session(impersonate="chrome124")
    response = session.get(_GENERATE, headers=_passport_headers(), timeout=20)
    response.raise_for_status()
    data = response.json().get("data") or {}
    if not data.get("url") or not data.get("qrcode_key"):
        raise RuntimeError(f"没有拿到登录二维码：{response.json()}")
    buffer = io.BytesIO()
    qrcode.make(data["url"]).save(buffer)
    return buffer.getvalue(), str(data["qrcode_key"])


def poll_login(qrcode_key: str, timeout: float = 180.0) -> None:
    """等到手机确认后把 Cookie 写入 COOKIE_PATH。"""
    deadline = time.time() + timeout
    seen = ""
    while time.time() < deadline:
        status = poll_once(qrcode_key)
        if status == LOGGED_IN:
            return
        if status == EXPIRED:
            raise RuntimeError("二维码已失效，请重新获取。")
        message = "已扫码，等待手机确认" if status == SCANNED else "等待扫码"
        if message != seen:
            print(message, flush=True)
            seen = message
        time.sleep(1.5)
    raise TimeoutError("登录超时，请重新获取二维码。")


# poll_once 的结果。
WAITING, SCANNED, LOGGED_IN, EXPIRED = "waiting", "scanned", "logged_in", "expired"


def poll_once(qrcode_key: str) -> str:
    """问一次扫码状态；手机确认了就把 Cookie 写入 COOKIE_PATH，返回 LOGGED_IN。"""
    from curl_cffi import requests

    session = requests.Session(impersonate="chrome124")
    response = session.get(_POLL.format(key=qrcode_key), headers=_passport_headers(), timeout=20)
    response.raise_for_status()
    body = response.json().get("data") or {}
    code = body.get("code")
    if code == 0:
        cookie = _merge_cookie(_cookie_header(session), _set_cookie_pairs(response))
        if "SESSDATA" not in cookie:
            raise RuntimeError("登录成功但没有 SESSDATA，请重新扫码。")
        COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
        COOKIE_PATH.write_text(cookie + "\n", encoding="utf-8")
        return LOGGED_IN
    if code == 86038:
        return EXPIRED
    return SCANNED if code == 86090 else WAITING


def logged_in() -> bool:
    return COOKIE_PATH.is_file() and "SESSDATA" in COOKIE_PATH.read_text(encoding="utf-8")


def logout() -> None:
    """退出登录：删掉保存的 Cookie。"""
    COOKIE_PATH.unlink(missing_ok=True)


def _passport_headers() -> dict[str, str]:
    return {
        "Referer": "https://www.bilibili.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
    }


def _set_cookie_pairs(response: object) -> str:
    headers = getattr(response, "headers", None)
    raw = getattr(headers, "raw", None)
    if not raw:
        return ""
    pairs: list[str] = []
    for key, value in raw:
        if key.decode("ascii", "ignore").lower() != "set-cookie" or not value:
            continue
        piece = value.decode("utf-8", "replace").split(";", 1)[0].strip()
        if "=" in piece:
            pairs.append(piece)
    return "; ".join(pairs)


def _merge_cookie(jar_cookie: str, set_cookie: str) -> str:
    found: dict[str, str] = {}
    for blob in (jar_cookie, set_cookie):
        for part in blob.split(";"):
            part = part.strip()
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            found[name.strip()] = value.strip()
    return "; ".join(f"{name}={value}" for name, value in found.items())


def _cookie_header(session: object) -> str:
    jar = getattr(session, "cookies", None)
    parts: list[str] = []
    items = getattr(jar, "items", None)
    if items is None:
        return ""
    for name, value in items():
        parts.append(f"{name}={value}")
    return "; ".join(parts)
