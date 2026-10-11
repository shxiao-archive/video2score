"""在"视频 + 标准答案"配对上跑一遍，看拼出来的小节全不全、干净图有没有丢音符、识别对不对。

python -m video2score bench-pdf                  全部（复用每个视频最近一次的作业）
python -m video2score bench-pdf --rerun          重新跑视频
python -m video2score bench-pdf <曲名> <曲名>     只跑这几首（jobs/_user_test 下的文件夹名）

测试集在 jobs/_user_test/<曲名>/：标准答案（MusicXML 优先，没有就用 Guitar Pro 导出的原谱 PDF），加上视频（source-1080.mp4 或
source.mp4；没有的话按 source.txt 里的 B 站链接下载成 source.mp4）。每首的明细写到作业目录的 compare.txt。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from video2score.compare import align_structure, compare, compare_xml
from video2score.inkcheck import check_job
from video2score.jobs import JOBS_DIR, load_manifest, save_manifest
from video2score.reference import read_reference
from video2score.run import Options, run, write_tab_score
from video2score.tabscore import read_job
from video2score.xmldiff import run_diff

TEST_DIR = JOBS_DIR / "_user_test"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    rerun = "--rerun" in args
    wanted = {arg for arg in args if not arg.startswith("--")}
    rows: list[str] = []
    for folder in sorted(TEST_DIR.iterdir()):
        if not folder.is_dir() or (wanted and folder.name not in wanted):
            continue
        # 标准答案：有 MusicXML 用 MusicXML（逐音校对过），没有再读原谱 PDF。
        answers = sorted(folder.glob("*.musicxml")) or sorted(folder.glob("*.pdf"))
        if not answers:
            continue
        started = time.time()
        try:
            video = _video(folder)
            job_dir = None if rerun else _latest_job(video)
            if job_dir is None:
                job_dir = run(str(video), Options(), progress=None)
            # 每次按现在的识别代码重新导出 MusicXML（不用重跑视频），再拿导出的文件和答案比。
            manifest = load_manifest(job_dir)
            manifest.setdefault("artifacts", {})
            write_tab_score(job_dir, manifest, title=folder.name)
            save_manifest(job_dir, manifest)
            measures = read_job(job_dir)
            reference = read_reference(answers[0])
            structure = align_structure(measures, reference)
            direct = None
            if answers[0].suffix.lower() == ".musicxml":
                result = compare_xml(job_dir / "score.musicxml", answers[0])
                lines = result.core.lines
                # 再直接对照两个 XML 的原文（双谱表、符杠、符干、闷音、拍号的写法），明细写 xmldiff.txt。
                direct = run_diff(job_dir / "score.musicxml", answers[0], job_dir / "xmldiff.txt")
            else:
                result = compare(measures, reference)
                lines = result.lines
            ink = check_job(job_dir)
        except Exception as exc:  # 一首出错不影响别的
            rows.append(f"{folder.name}: 出错 {exc}")
            print(rows[-1], flush=True)
            continue
        report = structure.summary() + "\n" + ink.summary() + "\n" + result.summary() + "\n\n" + "\n\n".join(lines) + "\n"
        (job_dir / "compare.txt").write_text(report, encoding="utf-8")
        rows.append(
            f"{folder.name}（{job_dir.name}，{time.time() - started:.0f} 秒）\n  {structure.summary()}\n  {ink.summary()}\n  "
            + result.summary().replace("\n", "\n  ")
            + ("\n  " + direct.summary().replace("\n", "\n  ") if direct is not None else "")
        )
        print(rows[-1], flush=True)
    return 0


def _video(folder: Path) -> Path:
    for name in ("source-1080.mp4", "source.mp4"):
        if (folder / name).is_file():
            return folder / name
    link = folder / "source.txt"
    if not link.is_file():
        raise FileNotFoundError(f"{folder} 里没有视频，也没有 source.txt")
    from video2score.fetch import download_bilibili

    return download_bilibili(link.read_text(encoding="utf-8").strip(), folder / "source.mp4")


def _latest_job(video: Path) -> Path | None:
    """这个视频最近一次跑完的作业。"""
    target = str(video.resolve())
    found = None
    for job_dir in sorted(JOBS_DIR.glob("2*")):
        try:
            manifest = load_manifest(job_dir)
        except Exception:
            continue
        if manifest.get("source", {}).get("ref") == target and (job_dir / "measures.json").is_file():
            found = job_dir
    return found


if __name__ == "__main__":
    raise SystemExit(main())
