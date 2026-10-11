"""历史作业：列表、详情、文件、有问题的小节、打开文件夹、删除（移到回收站，不真删）。"""

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
from video2score.paths import JOBS_DIR, TRASH_DIR

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
STATUS = {"captured": "只有截图谱", "omr_done": "完成", "failed": "失败"}


def job_dir(job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id or ""):
        raise FileNotFoundError(f"找不到作业：{job_id}")
    path = JOBS_DIR / job_id
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"找不到作业：{job_id}")
    return path


def list_jobs() -> list[dict[str, Any]]:
    """所有作业，新的在前。还在跑的作业（还没写 manifest.json）不列。"""
    if not JOBS_DIR.is_dir():
        return []
    found = []
    for path in sorted(JOBS_DIR.iterdir(), key=lambda item: item.name, reverse=True):
        if not _JOB_ID.fullmatch(path.name) or not (path / "manifest.json").is_file():
            continue
        try:
            found.append(summary(path, load_manifest(path)))
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
        "error": manifest.get("error"),
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


def remove(job_id: str) -> Path:
    """删除作业：整个目录移到回收站（trash/），需要时可以拿回来。"""
    path = job_dir(job_id)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    target = TRASH_DIR / path.name
    suffix = 1
    while target.exists():
        target = TRASH_DIR / f"{path.name}-{suffix}"
        suffix += 1
    shutil.move(str(path), str(target))
    return target


def _created(name: str) -> str:
    try:
        return datetime.strptime(name[:15], "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""
