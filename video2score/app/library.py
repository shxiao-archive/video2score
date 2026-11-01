"""历史作业：列表、详情、文件、有问题的小节、打开文件夹、删除（移到回收站）、清空回收站。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from video2score.jobs import load_manifest
from video2score.paths import JOBS_DIR, TRASH_DIR, UPLOADS_DIR

# 作业目录名：20261004-080439-1d78。jobs/ 下以 _ 开头的是测试集、回收站这类，不是作业。
_JOB_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")

# manifest 里的产物，按界面上的顺序；(名称, 下载时文件名的后缀)。
FILES = {
    "engraved_pdf": ("排版 PDF", ".pdf"),
    "pdf": ("截图谱 PDF", "（截图谱）.pdf"),
    "musicxml": ("MusicXML", ".musicxml"),
    "gp": ("GP（.gp）", ".gp"),
    "gpx": ("GP6（.gpx）", ".gpx"),
    "gp5": ("GP5（.gp5）", ".gp5"),
}
STATUS = {"captured": "只有截图谱", "omr_done": "完成", "failed": "失败", "running": "没跑完"}


def job_dir(job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id or ""):
        raise FileNotFoundError(f"找不到作业：{job_id}")
    path = JOBS_DIR / job_id
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"找不到作业：{job_id}")
    return path


def list_jobs() -> list[dict[str, Any]]:
    """所有作业，新的在前。本程序正在跑的不列（进度在"进行中"里）；别的 running 是跑到一半被杀的，列出来能删。

    shortcut: 只认本进程的 pid，命令行正在跑的作业这里也显示"没跑完"，要两边同时用再改成查进程是否活着。
    """
    if not JOBS_DIR.is_dir():
        return []
    found = []
    for path in sorted(JOBS_DIR.iterdir(), key=lambda item: item.name, reverse=True):
        if not _JOB_ID.fullmatch(path.name) or not (path / "manifest.json").is_file():
            continue
        try:
            manifest = load_manifest(path)
            if manifest.get("status") == "running" and manifest.get("pid") == os.getpid():
                continue
            found.append(summary(path, manifest))
        except Exception:  # 坏掉的 manifest 不影响别的作业
            continue
    return found


def summary(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from video2score.run import score_info

    title, artist, credit = score_info(manifest)
    video = manifest["source"].get("video") or {}
    artifacts = manifest.get("artifacts") or {}
    files = [
        {"key": key, "label": label, "path": artifacts[key]}
        for key, (label, _suffix) in FILES.items()
        if artifacts.get(key) and (path / artifacts[key]).is_file()
    ]
    return {
        "id": path.name,
        "title": title or path.name,
        "artist": artist,
        "credit": credit,
        "link": video.get("link") or (manifest["source"]["ref"] if manifest["source"]["type"] != "local" else ""),
        "created": _created(path.name),
        "status": manifest.get("status"),
        "status_text": STATUS.get(manifest.get("status"), manifest.get("status")),
        "error": manifest.get("error") or manifest.get("export_error"),
        "measures": manifest.get("measures"),
        "issues": len(issues(path)),
        "files": files,
        "can_export": (path / "score.musicxml").is_file(),
    }


def detail(job_id: str) -> dict[str, Any]:
    path = job_dir(job_id)
    data = summary(path, load_manifest(path))
    data["issue_list"] = issues(path)
    return data


def issues(path: Path) -> list[dict[str, Any]]:
    """有问题的小节：tab_issues.json（带小节图）；旧作业只有 tab_issues.txt，就按行列出。"""
    listed = path / "tab_issues.json"
    if listed.is_file():
        return json.loads(listed.read_text(encoding="utf-8"))
    text = path / "tab_issues.txt"
    if not text.is_file():
        return []
    return [{"number": None, "image": None, "issues": [line]} for line in text.read_text(encoding="utf-8").splitlines() if line.strip()]


def job_file(job_id: str, relative: str) -> Path:
    """作业目录里的一个文件；不许跳出作业目录。"""
    root = job_dir(job_id).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(relative)
    return path


def download_name(job_id: str, key: str) -> str:
    """下载时的文件名：曲名加后缀（去掉文件名里不能用的字符）。"""
    from video2score.run import score_info

    title = score_info(load_manifest(job_dir(job_id)))[0]
    title = re.sub(r'[\\/:*?"<>|\r\n]+', " ", title).strip()[:80] or job_id
    return title + FILES.get(key, ("", ""))[1]


def open_folder(job_id: str) -> None:
    path = job_dir(job_id)
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 —— 打开资源管理器
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _own_upload(path: Path) -> Path | None:
    """只有这个作业在用的上传视频；别的作业也在用（或者看不清）就返回 None。"""
    source = load_manifest(path)["source"]
    video = Path(source["ref"])
    if source["type"] != "local" or video.parent.resolve() != UPLOADS_DIR.resolve() or not video.is_file():
        return None
    for other in JOBS_DIR.iterdir():
        if other == path or not _JOB_ID.fullmatch(other.name) or not (other / "manifest.json").is_file():
            continue
        try:
            if load_manifest(other)["source"]["ref"] == source["ref"]:
                return None
        except Exception:
            return None
    return video


def remove(job_id: str, move_upload: bool = False) -> Path:
    """删除作业：整个目录移到回收站（trash/），需要时可以拿回来。move_upload：拖进来的视频没别人用，也一起移过去。"""
    path = job_dir(job_id)
    upload = _own_upload(path) if move_upload else None
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    target = TRASH_DIR / path.name
    suffix = 1
    while target.exists():
        target = TRASH_DIR / f"{path.name}-{suffix}"
        suffix += 1
    shutil.move(str(path), str(target))
    if upload is not None:
        (target / "upload").mkdir()
        shutil.move(str(upload), str(target / "upload" / upload.name))
    return target


def empty_trash() -> int:
    """真删回收站里的东西，返回腾出了多少字节。"""
    if not TRASH_DIR.is_dir():
        return 0
    freed = 0
    for item in TRASH_DIR.iterdir():
        if item.is_dir():
            freed += sum(file.stat().st_size for file in item.rglob("*") if file.is_file())
            shutil.rmtree(item)
        else:
            freed += item.stat().st_size
            item.unlink()
    return freed


def _created(name: str) -> str:
    try:
        return datetime.strptime(name[:15], "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""
