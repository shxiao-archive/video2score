"""本机网页界面的服务端（只用标准库）。

`python -m video2score app` 或双击打包好的 video2score：在 127.0.0.1 上起服务，自动打开浏览器。
- 只监听本机；每次启动生成一个随机令牌，接口都要带（页面里带着），并检查 Host 头，别的网页借不了本机端口。
- 转谱、补导出交给 worker 排队；作业、上传的视频、回收站、日志都在 paths 定的数据目录里。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import re
import secrets
import shutil
import socket
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from video2score import __version__, paths
from video2score.app import library
from video2score.app.worker import Task, Worker

log = logging.getLogger("video2score")

PREFERRED_PORT = 17865
EXPORTS = ("gp", "gpx", "gp5", "pdf")
_UPLOAD_CHUNK = 1 << 20


class App:
    def __init__(self, port: int | None = None) -> None:
        self.token = secrets.token_urlsafe(24)
        self.worker = Worker()
        self.server = ThreadingHTTPServer(("127.0.0.1", _free_port(port)), Handler)
        self.server.daemon_threads = True
        self.server.app = self  # type: ignore[attr-defined]
        self.port = self.server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def serve(self) -> None:
        self.server.serve_forever()

    def stop(self) -> None:
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    # ------------------------------------------------------------ 状态

    def state(self) -> dict[str, Any]:
        from video2score.engrave import find_browser
        from video2score.login import logged_in

        return {
            "version": __version__,
            "logged_in": logged_in(),
            "browser": find_browser() is not None,
            "node": paths.node() is not None,
            "translocated": paths.translocated(),
            "app_root": str(paths.APP_ROOT),
            "data_dir": str(paths.DATA_DIR),
            "tasks": self.worker.tasks(),
        }

    # ------------------------------------------------------------ 任务

    def convert(self, source: str, link: str | None, exports: list[str], label: str | None = None) -> Task:
        """转谱：先跑识别（截图谱、MusicXML），再按勾选导出；导出失败只记一句，作业照样算完成。"""
        from video2score.run import Options, run

        source = source.strip()
        if not source:
            raise ValueError("请粘贴视频链接，或拖入一个视频文件。")
        wanted = self._exports(exports)

        def work(task: Task) -> str:
            job = run(source, Options(link=link or None), progress=task.say)
            task.job = job.name
            self._export(task, job, wanted)
            return job.name

        return self.worker.submit(Task("convert", label or source, work))

    def export(self, job_id: str, exports: list[str]) -> Task:
        job = library.job_dir(job_id)
        wanted = self._exports(exports)
        if not wanted:
            raise ValueError("请选择要导出的格式。")

        def work(task: Task) -> str:
            task.say("导出 " + "、".join(wanted) + "……")
            self._export(task, job, wanted, strict=True)
            return job.name

        task = Task("export", library.summary(job, _manifest(job))["title"], work)
        task.job = job.name
        return self.worker.submit(task)

    def _exports(self, exports: list[str]) -> list[str]:
        unknown = [item for item in exports if item not in EXPORTS]
        if unknown:
            raise ValueError(f"不支持的导出格式：{', '.join(unknown)}")
        return [item for item in EXPORTS if item in exports]

    @staticmethod
    def _export(task: Task, job: Path, wanted: list[str], strict: bool = False) -> None:
        from video2score.jobs import save_manifest
        from video2score.run import export_score

        manifest = _manifest(job)
        if not wanted:
            return
        if not (job / "score.musicxml").is_file():
            task.error = "这首只认出了五线谱（没有 TAB），还不能导出 MusicXML 和 Guitar Pro 文件。"
            return
        task.say("导出 " + "、".join(wanted) + "……")
        try:
            export_score(job, manifest, wanted)
        except Exception as exc:
            if strict:
                raise
            task.error = f"导出没完成：{exc}"
        finally:
            save_manifest(job, manifest)

    def upload(self, name: str, size: int, stream) -> Path:
        """拖进来的视频存到 uploads/（同名的加序号），按本地视频转谱。"""
        name = re.sub(r'[\\/:*?"<>|\r\n]+', "_", Path(name).name).strip() or "video.mp4"
        paths.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        target = paths.UPLOADS_DIR / name
        stem, suffix, index = target.stem, target.suffix, 1
        while target.exists():
            target = paths.UPLOADS_DIR / f"{stem}-{index}{suffix}"
            index += 1
        left = size
        with target.open("wb") as handle:
            while left > 0:
                chunk = stream.read(min(_UPLOAD_CHUNK, left))
                if not chunk:
                    break
                handle.write(chunk)
                left -= len(chunk)
        if left:
            target.unlink(missing_ok=True)
            raise ValueError("上传没有完成，请重试。")
        return target


def _manifest(job: Path) -> dict[str, Any]:
    from video2score.jobs import load_manifest

    return load_manifest(job)


def _free_port(port: int | None) -> int:
    for candidate in ([port] if port else [PREFERRED_PORT, 0]):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
                return probe.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError(f"端口 {port} 已被占用。")


class Handler(BaseHTTPRequestHandler):
    server_version = "video2score"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 —— 沿用基类的参数名
        log.debug("%s %s", self.address_string(), format % args)

    # ------------------------------------------------------------ 入口

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        self.query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        if self.headers.get("Host") not in (f"127.0.0.1:{self.app.port}", f"localhost:{self.app.port}"):
            return self._error(HTTPStatus.FORBIDDEN, "只接受本机访问。")
        try:
            if method == "GET" and url.path in ("/", "/index.html"):
                return self._index()
            if method == "GET" and url.path.startswith("/static/"):
                return self._static(url.path[len("/static/") :])
            if not url.path.startswith("/api/"):
                return self._error(HTTPStatus.NOT_FOUND, "没有这个页面。")
            if not secrets.compare_digest(self.headers.get("X-Token") or self.query.get("t", ""), self.app.token):
                return self._error(HTTPStatus.FORBIDDEN, "令牌不对，请从 video2score 打开的页面操作。")
            route = getattr(self, f"_{method.lower()}_" + url.path[len("/api/") :].split("/")[0], None)
            if route is None:
                return self._error(HTTPStatus.NOT_FOUND, "没有这个接口。")
            route([part for part in url.path[len("/api/") :].split("/")[1:] if part])
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc) or "找不到。")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            log.exception("处理 %s %s 出错", method, url.path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    # ------------------------------------------------------------ 页面

    def _index(self) -> None:
        page = (paths.WEB / "index.html").read_text(encoding="utf-8")
        page = page.replace("__TOKEN__", self.app.token).replace("__VERSION__", __version__)
        self._send(page.encode("utf-8"), "text/html; charset=utf-8", cache=False)

    def _static(self, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not (paths.WEB / name).is_file():
            return self._error(HTTPStatus.NOT_FOUND, "没有这个文件。")
        kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind.endswith("javascript"):
            kind += "; charset=utf-8"
        self._send((paths.WEB / name).read_bytes(), kind, cache=False)

    # ------------------------------------------------------------ 接口

    def _get_state(self, _parts: list[str]) -> None:
        self._json(self.app.state())

    def _get_jobs(self, parts: list[str]) -> None:
        if not parts:
            return self._json(library.list_jobs())
        if len(parts) == 1:
            return self._json(library.detail(parts[0]))
        # /api/jobs/<id>/file?path=...&key=...&download=1
        if parts[1] == "file":
            path = library.job_file(parts[0], self.query.get("path", ""))
            name = library.download_name(parts[0], self.query["key"]) if self.query.get("key") else path.name
            return self._file(path, name, attachment=self.query.get("download") == "1")
        self._error(HTTPStatus.NOT_FOUND, "没有这个接口。")

    def _post_convert(self, _parts: list[str]) -> None:
        body = self._body()
        task = self.app.convert(str(body.get("source") or ""), body.get("link"), list(body.get("exports") or []), body.get("label"))
        self._json(task.to_dict())

    def _post_upload(self, _parts: list[str]) -> None:
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0:
            raise ValueError("没有收到文件。")
        path = self.app.upload(self.query.get("name", "video.mp4"), size, self.rfile)
        self._json({"path": str(path), "name": path.name})

    def _post_jobs(self, parts: list[str]) -> None:
        if len(parts) != 2:
            return self._error(HTTPStatus.NOT_FOUND, "没有这个接口。")
        job_id, action = parts
        if action == "export":
            return self._json(self.app.export(job_id, list(self._body().get("exports") or [])).to_dict())
        if action == "open":
            library.open_folder(job_id)
            return self._json({"ok": True})
        if action == "remove":
            if any(task["job"] == job_id and task["state"] in ("queued", "running") for task in self.app.worker.tasks()):
                raise ValueError("这首还在导出，等它做完再删。")
            target = library.remove(job_id)
            return self._json({"ok": True, "trash": str(target)})
        self._error(HTTPStatus.NOT_FOUND, "没有这个接口。")

    def _post_tasks(self, parts: list[str]) -> None:
        if len(parts) == 2 and parts[1] == "cancel":
            return self._json({"ok": self.app.worker.cancel(parts[0])})
        self._error(HTTPStatus.NOT_FOUND, "没有这个接口。")

    def _post_login(self, _parts: list[str]) -> None:
        from video2score.login import login_qr

        png, key = login_qr()
        self._json({"key": key, "qr": "data:image/png;base64," + base64.b64encode(png).decode("ascii")})

    def _get_login(self, _parts: list[str]) -> None:
        from video2score.login import poll_once

        try:
            self._json({"status": poll_once(self.query.get("key", ""))})
        except Exception as exc:
            self._json({"status": "error", "error": str(exc)})

    def _post_logout(self, _parts: list[str]) -> None:
        from video2score.login import logout

        logout()
        self._json({"ok": True})

    def _post_quit(self, _parts: list[str]) -> None:
        self._json({"ok": True})
        log.info("界面上点了退出")
        self.app.stop()

    # ------------------------------------------------------------ 回应

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length") or 0)
        if not size:
            return {}
        data = json.loads(self.rfile.read(size).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求格式不对。")
        return data

    def _json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status, cache=False)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _file(self, path: Path, name: str, attachment: bool) -> None:
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        disposition = "attachment" if attachment else "inline"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f"{disposition}; filename*=UTF-8''{quote(name)}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _send(self, data: bytes, kind: str, status: HTTPStatus = HTTPStatus.OK, cache: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def setup_logging() -> None:
    paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [RotatingFileHandler(paths.LOGS_DIR / "app.log", maxBytes=1 << 20, backupCount=3, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def main(argv: list[str] | None = None) -> int:
    # 英文等系统上控制台、管道默认不是 UTF-8，打印中文会出错：一律按 UTF-8 输出，写不了的字换掉。
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="video2score", description="视频转谱：本机网页界面")
    parser.add_argument("--port", type=int, default=None, help=f"端口，默认 {PREFERRED_PORT}（被占用时随便找一个空的）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--selftest", action="store_true", help="自检：起服务、调接口、导出一份小谱，然后退出")
    # macOS 从访达打开 .app 时可能带上系统自己的参数（-psn_…），不认识的忽略。
    args, _unknown = parser.parse_known_args(argv)
    setup_logging()
    if args.selftest:
        from video2score.app.selftest import selftest

        return selftest()
    app = App(args.port)
    log.info("video2score %s：%s（数据在 %s）", __version__, app.url, paths.DATA_DIR)
    if paths.FROZEN and sys.stderr is not None:
        print(f"\n  视频转谱已启动：{app.url}\n  关掉这个窗口或在页面上点\"退出\"就会停止。\n", file=sys.stderr, flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(app.url,)).start()
    try:
        app.serve()
    except KeyboardInterrupt:
        pass
    return 0
