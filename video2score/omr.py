"""五线谱交给 Audiveris 或 oemer。这一轮还没接进 run，留给下一轮接到小节库上。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from video2score.jobs import (
    VENDOR_OEMER,
    load_manifest,
    relative_to_job,
    resolve_job_dir,
    save_manifest,
)


def transcribe(job_id: str, backend: str = "auto") -> Path:
    job_dir = resolve_job_dir(job_id)
    manifest = load_manifest(job_dir)
    if manifest.get("notation") != "staff":
        raise RuntimeError("Audiveris 和 oemer 只认五线谱，不读 TAB 的品位数字。")

    pdf_path = job_dir / str((manifest.get("artifacts") or {}).get("pdf") or "score.pdf")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"作业里没有 PDF：{pdf_path}")

    chosen = _select_backend(backend)
    try:
        if chosen == "audiveris":
            produced = _run_audiveris(job_dir, pdf_path)
        else:
            produced = _run_oemer(job_dir, pdf_path)
        if not produced:
            raise RuntimeError(f"{chosen} 没有产出 MusicXML。")
        recorded = [relative_to_job(job_dir, path) for path in produced]
        manifest["artifacts"]["musicxml"] = recorded[0] if len(recorded) == 1 else recorded
        manifest["omr"] = {"backend": chosen}
        manifest["status"] = "omr_done"
        manifest.pop("error", None)
        save_manifest(job_dir, manifest)
        return job_dir
    except Exception as exc:
        manifest["error"] = str(exc)
        manifest["status"] = "captured"
        manifest["artifacts"].pop("musicxml", None)
        save_manifest(job_dir, manifest)
        raise


def _select_backend(backend: str) -> str:
    if backend not in {"auto", "audiveris", "oemer"}:
        raise ValueError("omr 后端只能是 auto、audiveris 或 oemer。")
    if backend == "oemer":
        return "oemer"
    found = shutil.which("audiveris")
    if backend == "audiveris":
        if not found:
            raise RuntimeError("PATH 中没有 audiveris。")
        return "audiveris"
    return "audiveris" if found else "oemer"


def _run_audiveris(job_dir: Path, pdf_path: Path) -> list[Path]:
    executable = shutil.which("audiveris")
    if not executable:
        raise RuntimeError("PATH 中没有 audiveris。")
    output_dir = job_dir / "musicxml"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-batch",
        "-transcribe",
        "-export",
        "-output",
        str(output_dir),
        str(pdf_path),
    ]
    completed = subprocess.run(
        command,
        cwd=job_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    produced = _musicxml_files(output_dir)
    if completed.returncode != 0 and not produced:
        raise RuntimeError(_process_error("Audiveris", completed))
    return produced


def _run_oemer(job_dir: Path, pdf_path: Path) -> list[Path]:
    if not (VENDOR_OEMER / "oemer").is_dir():
        raise RuntimeError(f"找不到 oemer：{VENDOR_OEMER}")
    pages_dir = job_dir / "pages"
    page_paths = _render_pages(pdf_path, pages_dir)
    output_dir = job_dir / "musicxml"
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for page_path in page_paths:
        before = set(_musicxml_files(output_dir))
        completed = subprocess.run(
            [sys.executable, "-m", "oemer.ete", str(page_path), "-o", str(output_dir)],
            cwd=job_dir,
            env=_oemer_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        new_files = [path for path in _musicxml_files(output_dir) if path not in before]
        if completed.returncode != 0 or not new_files:
            detail = _process_error("oemer", completed)
            raise RuntimeError(f"{page_path.name}：{detail}")
        produced.extend(new_files)
    return produced


def _render_pages(pdf_path: Path, pages_dir: Path) -> list[Path]:
    """每页渲染成 200 dpi 的 PNG，交给只吃单张图的 oemer。"""
    import pymupdf

    pages_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    document = pymupdf.open(pdf_path)
    try:
        for index, page in enumerate(document, start=1):
            path = pages_dir / f"page_{index:04d}.png"
            page.get_pixmap(dpi=200).save(path)
            paths.append(path)
    finally:
        document.close()
    if not paths:
        raise RuntimeError("PDF 没有可渲染的页面。")
    return paths


def _oemer_env() -> dict[str, str]:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    root = str(VENDOR_OEMER)
    env["PYTHONPATH"] = root if not previous else root + os.pathsep + previous
    return env


def _musicxml_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    found = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".musicxml", ".mxl", ".xml"}
    ]
    return sorted(found)


def _process_error(name: str, completed: subprocess.CompletedProcess[str]) -> str:
    text = (completed.stderr or completed.stdout or "").strip()
    if len(text) > 2000:
        text = text[-2000:]
    if not text:
        text = f"退出码 {completed.returncode}"
    return f"{name} 失败：{text}"
