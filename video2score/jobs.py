"""作业目录与 manifest.json。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from video2score.paths import APP_ROOT, JOBS_DIR

ROOT = APP_ROOT
VENDOR_OEMER = ROOT / "vendor" / "oemer"

MANIFEST_VERSION = 2
SOURCE_TYPES = {"local", "bilibili", "url"}
NOTATIONS = {"staff", "tab", "staff+tab"}
STATUSES = {"captured", "failed", "omr_done"}


def new_job_dir() -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        path = JOBS_DIR / job_id
        if not path.exists():
            path.mkdir()
            return path
    raise RuntimeError("无法创建作业目录。")


def resolve_job_dir(job_id: str) -> Path:
    direct = Path(job_id)
    if direct.is_dir() and (direct / "manifest.json").is_file():
        return direct.resolve()
    candidate = JOBS_DIR / job_id
    if candidate.is_dir() and (candidate / "manifest.json").is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"找不到作业：{job_id}")


def manifest_path(job_dir: Path) -> Path:
    return job_dir / "manifest.json"


def load_manifest(job_dir: Path) -> dict[str, Any]:
    path = manifest_path(job_dir)
    if not path.is_file():
        raise FileNotFoundError(f"作业缺少 manifest.json：{job_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(job_dir: Path, data: dict[str, Any], *, strict: bool = True) -> None:
    if strict:
        validate_manifest(data)
    manifest_path(job_dir).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def relative_to_job(job_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(job_dir.resolve()).as_posix()


def validate_manifest(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("manifest 必须是对象。")
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"manifest.version 必须是 {MANIFEST_VERSION}。")

    source = data.get("source")
    if not isinstance(source, dict) or source.get("type") not in SOURCE_TYPES:
        raise ValueError("source.type 无效。")
    if not str(source.get("ref") or "").strip():
        raise ValueError("source.ref 不能为空。")

    status = data.get("status")
    if status not in STATUSES:
        raise ValueError("status 无效。")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts 必须是对象。")
    if status == "failed":
        if not str(data.get("error") or "").strip():
            raise ValueError("failed 需要写明 error。")
        return

    if data.get("notation") not in NOTATIONS:
        raise ValueError("notation 无效。")
    for key in ("pdf", "library"):
        if not str(artifacts.get(key) or "").strip():
            raise ValueError(f"缺少 artifacts.{key}。")
    measures = data.get("measures")
    if not isinstance(measures, int) or measures < 1:
        raise ValueError("measures 必须是正整数。")

    musicxml = artifacts.get("musicxml")
    if status != "omr_done" and musicxml:
        raise ValueError("只有 omr_done 可以带 musicxml。")
    if status == "omr_done":
        _require_musicxml(musicxml)


def _require_musicxml(musicxml: Any) -> None:
    if isinstance(musicxml, str):
        if not musicxml.strip():
            raise ValueError("musicxml 为空。")
        return
    if isinstance(musicxml, list) and musicxml and all(
        isinstance(item, str) and item.strip() for item in musicxml
    ):
        return
    raise ValueError("musicxml 必须是字符串或非空字符串数组。")
