"""打包后的自检（`video2score --selftest`，CI 构建完跑）：起服务、调接口，在中文目录里把一段合成视频转成谱，
再把一份合成的小谱导出成 Guitar Pro 文件和排版 PDF。

用到的是包里自带的东西：网页、GP 模板、冻结的字形模板、Node、alphaTab、Puppeteer 和 chrome-headless-shell。
打包后的程序一定要导得出 PDF；从仓库跑自检时没有浏览器才跳过。
中间文件放在数据目录下的 selftest/中文目录/（路径问题要在这里暴露），做完删掉。
"""

from __future__ import annotations

import contextlib
import json
import shutil
import threading
import urllib.request
from fractions import Fraction
from pathlib import Path

from video2score import jobs, paths


def _convert_video(folder: Path) -> None:
    """合成一段滚动的 TAB 视频，走一遍转谱主流程。"""
    import cv2

    from video2score.run import Options, run
    from video2score.staffdraw import band_frame, draw_tab_strip

    riff = ["4", "4", "4", "4"]
    strip = draw_tab_strip([riff, ["5", "7"], riff, ["2", "2", "3"], riff, ["9"]], measure_width=180)
    last = strip.shape[1] - 640
    # cv2.VideoWriter 写不了中文路径（读可以）：切进目录写相对文件名。
    with contextlib.chdir(folder):
        writer = cv2.VideoWriter("clip.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
        for offset in [0] * 12 + list(range(0, last, 12)) + [last] * 12:
            writer.write(cv2.cvtColor(band_frame(strip, offset), cv2.COLOR_RGB2BGR))
        writer.release()
    jobs_dir = jobs.JOBS_DIR
    jobs.JOBS_DIR = folder / "jobs"
    try:
        job = run(str(folder / "clip.avi"), Options(sample_every=0.3))
    finally:
        jobs.JOBS_DIR = jobs_dir
    manifest = jobs.load_manifest(job)
    assert (manifest["status"], manifest["notation"], manifest["measures"]) == ("omr_done", "tab", 6), manifest
    assert (job / "score.musicxml").is_file(), job


def selftest() -> int:
    from video2score.app.server import App
    from video2score.engrave import find_browser, musicxml_to_pdf
    from video2score.glyphs import frozen_bank
    from video2score.gpfile import export
    from video2score.tab import FretNote
    from video2score.tabscore import Marks, ScoreMeasure, ScoreNote, write_musicxml

    checks: list[str] = []
    app = App()
    threading.Thread(target=app.serve, daemon=True).start()
    try:
        request = urllib.request.Request(app.url + "api/state", headers={"X-Token": app.token})
        with urllib.request.urlopen(request, timeout=10) as response:
            state = json.loads(response.read().decode("utf-8"))
        checks.append(f"服务 {state['version']}")
        with urllib.request.urlopen(app.url, timeout=10) as response:
            assert app.token in response.read().decode("utf-8")
        checks.append("页面")
    finally:
        app.server.shutdown()
    assert frozen_bank("tab") is not None, "缺少冻结的字形模板 glyphs.npz"
    checks.append("字形模板")
    folder = paths.DATA_DIR / "selftest" / "中文目录"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        _convert_video(folder)
        checks.append("转谱主流程（中文目录）")
        note = ScoreNote(Fraction(1, 4), frets=[FretNote(3, 2, 0, 0, 0, 1.0)], kind="quarter")
        notes = [note] + [ScoreNote(Fraction(1, 4), frets=[FretNote(2, fret, 0, 0, 0, 1.0)], kind="quarter") for fret in (0, 2, 3)]
        source = folder / "selftest.musicxml"
        write_musicxml(source, [ScoreMeasure(1, notes, Fraction(1), 4)], title="自检", marks=Marks(tempo=120), rights="视频：自检 https://example.com/")
        for path in export(source, ["gp", "gpx", "gp5"], folder, "selftest"):
            assert path.stat().st_size > 0, path
        checks.append("Guitar Pro 导出")
        if paths.FROZEN:
            assert paths.BUNDLED_BROWSER.is_file(), f"包里缺少 chrome-headless-shell：{paths.BUNDLED_BROWSER}"
        if find_browser() is not None:
            pdf = musicxml_to_pdf(source, folder / "selftest.pdf")
            assert pdf.stat().st_size > 1000, pdf
            checks.append("排版 PDF")
        else:
            checks.append("排版 PDF（跳过：没有浏览器）")
    finally:
        shutil.rmtree(folder.parent, ignore_errors=True)
    print("自检通过：" + "、".join(checks), flush=True)
    return 0
