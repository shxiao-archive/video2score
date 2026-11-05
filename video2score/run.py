"""一次转换：取视频 → 逐帧切小节 → 追踪去重 → 清洗 → 重新排版成 PDF。"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from video2score.images import read_gray, write_image
from video2score.clean import CleanMeasure, render_prefix, render_slot
from video2score.frames import iter_frames, video_info
from video2score.header import Header, content_start, erase_label_after_header, extract_header
from video2score.tidy import geometry, has_multirest
from video2score.jobs import Cancelled, MANIFEST_VERSION, check_formats, load_manifest, new_job_dir, relative_to_job, save_manifest
from video2score.layout import layout_rows, paginate, write_pdf
from video2score.measures import PAD_ABOVE, MeasureFinder
from video2score.paths import UPLOADS_DIR
from video2score.order import arrange, gaps, rescue
from video2score.tabscore import ScoreMeasure, estimate_tempo, read_job, score_marks, write_musicxml
from video2score.tabscore import summary as tab_summary
from video2score.source import classify_source, fetch_video
from video2score.survey import survey
from video2score.track import Slot, Tracker, fragments, ordered_slots, summary
from video2score.videoinfo import VideoInfo, lookup

# 取帧间隔。要让相邻两帧之间谱面的位移小于半个小节。
DEFAULT_SAMPLE_EVERY = 0.33


@dataclass
class Options:
    notation: str | None = None
    start_sec: float = 0.0
    end_sec: float | None = None
    sample_every: float = DEFAULT_SAMPLE_EVERY
    ratios: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None)
    debug: bool = False
    # 除了 score.musicxml 另外导出的格式："gp"、"gpx"、"gp5"、"pdf"（按识别结果重新排版的谱）。
    exports: tuple[str, ...] = ()
    # 本地视频的原视频链接：用来查曲名、原唱，页脚印上出处。
    link: str | None = None


def run(
    source: str,
    options: Options | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    options = options or Options()
    check_formats(options.exports)
    say = progress or (lambda _text: None)
    kind, ref = classify_source(source)
    job_dir = new_job_dir()
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "source": {"type": kind, "ref": ref},
        "status": "running",
        "pid": os.getpid(),
        "artifacts": {},
    }
    # 一开始就写下来：进程半路被杀，界面上还能看见这个作业。
    save_manifest(job_dir, manifest, strict=False)
    try:
        if is_pdf(kind, ref):
            _run_pdf(job_dir, manifest, Path(ref), options, say)
        else:
            _run_video(job_dir, manifest, source, kind, options, say)
        save_manifest(job_dir, manifest)
        return job_dir
    except Cancelled:
        # 用户不要了：不留半截作业。
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest.pop("pid", None)
        save_manifest(job_dir, manifest, strict=False)
        raise


def is_pdf(kind: str, ref: str) -> bool:
    return kind == "local" and Path(ref).suffix.lower() == ".pdf"


def _run_video(job_dir: Path, manifest: dict[str, Any], source: str, kind: str, options: Options, say: Callable[[str], None]) -> None:
    say("取视频……")
    video = fetch_video(source, kind, job_dir / "download", say)
    video_of(manifest, options.link, say)
    fps, frame_count = video_info(video)
    say("看谱面是哪一种……")
    mode = survey(video, start_sec=options.start_sec, end_sec=options.end_sec, ratios=options.ratios)
    say(f"  {mode.describe()}")
    tracker = Tracker()
    finder = MeasureFinder(options.ratios, mode.scale, mode.layout == "page")
    sampled = 0
    say("逐帧找小节……")
    # closing：半路取消时马上关掉视频文件，不然 Windows 上删不掉作业目录。
    with contextlib.closing(iter_frames(video, sample_every=options.sample_every, start_sec=options.start_sec, end_sec=options.end_sec)) as frames:
        for frame in frames:
            sampled += 1
            found = finder(frame)
            if found is not None:
                tracker.add(found)
            if sampled % 100 == 0 and frame_count:
                say(f"  {frame.index / frame_count:.0%}  已收 {len(tracker.slots)} 个小节")
    manifest["sampling"] = {
        "every": options.sample_every,
        "frames": sampled,
        "fps": round(fps, 3),
        "polarity": finder.polarity,
        "mode": mode.to_json(),
        "transitions": summary(tracker),
    }
    slots = ordered_slots(tracker)
    if not slots:
        raise RuntimeError("没有找到完整的小节。请确认画面里有谱线和小节线，或用 --crop-* 指定谱面区域。")
    say("读小节号……")
    slots, numbering = arrange(slots)
    # 号断了档的地方，到只出现过一两帧的碎片里找回来。
    slots, rescued = rescue(slots, fragments(tracker))
    manifest["numbers"] = {
        "read": numbering.read,
        "total": numbering.total,
        "moved": numbering.moved,
        "merged": numbering.merged,
        "split": numbering.split,
        "cleared": numbering.cleared,
        "rescued": rescued,
        "jumps": gaps(slots),
    }
    _finish(job_dir, manifest, slots, options, say)
    if options.debug:
        _write_debug(job_dir, tracker, slots)


def _run_pdf(job_dir: Path, manifest: dict[str, Any], pdf: Path, options: Options, say: Callable[[str], None]) -> None:
    """PDF：只收 Guitar Pro 导出的矢量谱，直接读文字和线条。截图拼的、扫描的按图识别不如转视频，不做。"""
    from video2score.vectorpdf import is_gp_vector

    if not pdf.is_file():
        raise FileNotFoundError(f"找不到 {pdf}")
    if not is_gp_vector(pdf):
        raise RuntimeError("只支持 Guitar Pro 导出的 PDF 谱。截图谱、扫描件请改用原视频转。")
    video_of(manifest, options.link, say)
    _run_vector(job_dir, manifest, pdf, options, say)


def _run_vector(job_dir: Path, manifest: dict[str, Any], pdf: Path, options: Options, say: Callable[[str], None]) -> None:
    """GP 矢量谱：score.pdf 就是原 PDF；小节库只记每小节的号、页、方框（没有识别问题要配图，不裁小节图）。"""
    from video2score.reference import read_reference

    say("读矢量谱……")
    measures = read_reference(pdf)
    if not measures:
        raise RuntimeError("PDF 里没有读到小节。")
    shutil.copyfile(pdf, job_dir / "score.pdf")
    records = [{"index": index, "number": measure.number, "page": measure.page, "box": [round(value, 1) for value in measure.box or ()]} for index, measure in enumerate(measures, start=1)]
    library = job_dir / "measures.json"
    library.write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest["pages"] = max(measure.page for measure in measures) + 1
    manifest["vector"] = True
    manifest["notation"] = options.notation or ("staff+tab" if any(measure.strings for measure in measures) else "staff")
    manifest["measures"] = len(measures)
    manifest["artifacts"] = {"pdf": "score.pdf", "library": relative_to_job(job_dir, library)}
    manifest["status"] = "captured"
    manifest.pop("pid")
    _tab_and_export(job_dir, manifest, options, say)


def _finish(job_dir: Path, manifest: dict[str, Any], slots: list[Slot], options: Options, say: Callable[[str], None]) -> None:
    """排好序、定了号的小节 → 清洗、小节库、截图谱 PDF，再读 TAB、导出。"""
    title, artist, credit = score_info(manifest)
    cleaned = []
    for index, slot in enumerate(slots):
        if index % 10 == 0:
            say(f"清洗 {index}/{len(slots)}……")
        cleaned.append(render_slot(slot))
    library = _save_library(job_dir, slots, cleaned)
    # 乐谱最开头第一小节左边的乐器名、调弦字母。
    prefix = render_prefix(slots)
    if prefix is not None:
        write_image(job_dir / _PREFIX, prefix)
    say("排版……")
    pdf_path = job_dir / "score.pdf"
    pages = write_score_pdf(pdf_path, cleaned, [slot.number for slot in slots], title, prefix, artist, credit)
    manifest["notation"] = options.notation or _notation(cleaned)
    manifest["measures"] = len(slots)
    manifest["pages"] = len(pages)
    manifest["artifacts"] = {
        "pdf": relative_to_job(job_dir, pdf_path),
        "library": relative_to_job(job_dir, library),
        "measures": "measures",
    }
    if prefix is not None:
        manifest["artifacts"]["prefix"] = _PREFIX
    manifest["status"] = "captured"
    manifest.pop("pid")
    _tab_and_export(job_dir, manifest, options, say)


def _tab_and_export(job_dir: Path, manifest: dict[str, Any], options: Options, say: Callable[[str], None]) -> None:
    # 截图谱已经有了：后面哪一步出错都不算整个作业失败，记一句，界面和命令行都会提示。
    if "tab" not in manifest["notation"]:
        return
    try:
        say("读 TAB 和节奏……")
        write_tab_score(job_dir, manifest, progress=say)
    except Exception as exc:
        manifest["error"] = f"读 TAB 没完成：{exc}"
        return
    if options.exports:
        say("导出 " + "、".join(options.exports) + "……")
        try:
            export_score(job_dir, manifest, options.exports)
        except Exception as exc:
            manifest["export_error"] = f"导出没完成：{exc}"


def skipped_measures(measures: list[ScoreMeasure]) -> list[tuple[int, int]]:
    """号断了档的地方：(从第几小节, 到第几小节) 视频里没读到，谱里缺了。合并休止占的号不算。"""
    numbered = [(index, measure) for index, measure in enumerate(measures) if measure.number is not None]
    skipped = []
    for position, (index, measure) in enumerate(numbered[:-1]):
        following_index, following = numbered[position + 1]
        between = measures[index:following_index]
        expected = measure.number + sum(1 + item.rests_after for item in between)
        # 整小节休止格后面的断档，是没展开的合并休止，它自己已经列为问题了。
        if following.number <= expected or any(all(note.rest for note in item.notes) for item in between):
            continue
        # 号可能读错（见过读成 511 的）：后面还有号的，得比它大；已经是最后一个号时，只信小的间隔。
        after = numbered[position + 2][1].number if position + 2 < len(numbered) else None
        if (after is not None and after > following.number) or (after is None and following.number - expected <= 16):
            skipped.append((expected, following.number - 1))
    return skipped


def write_tab_score(job_dir: Path, manifest: dict[str, Any], title: str | None = None, progress: Callable[[str], None] | None = None) -> None:
    """读小节库里的 TAB 和节奏，写 score.musicxml；有问题的小节列进 tab_issues.txt（界面用的 tab_issues.json 带小节图）。

    曲名、原唱、出处（视频作者和链接）按作业的视频信息写；给了 title 就用它当曲名。
    """
    found, artist, credit = score_info(manifest)
    path = job_dir / "score.musicxml"
    where = "PDF" if manifest.get("vector") else "视频"
    if manifest.get("vector"):
        # GP 矢量谱：重新读一遍 score.pdf（就是原 PDF），速度、调弦、演奏法都从矢量里来。
        from video2score.vectorpdf import write_score

        if progress:
            progress("读矢量谱……")
        measures, tempo = write_score(job_dir / "score.pdf", path, title or found, artist, credit)
        tempo_text = [] if tempo else ["速度：PDF 里没印"]
    else:
        measures = read_job(job_dir, progress)
        marks = score_marks(job_dir)
        if marks.tempo is None:
            # 画面里没印速度：按每格出现在画面上的时间估一个。
            marks.tempo = estimate_tempo(measures)
            marks.tempo_estimated = marks.tempo is not None
        write_musicxml(path, measures, title=title or found, marks=marks, artist=artist, rights=credit)
        tempo_text = [f"速度：画面里没印，按视频时间估成 {marks.tempo}"] if marks.tempo_estimated else []
    issues = [
        f"第 {measure.number if measure.number is not None else '?'} 小节：{issue}"
        for measure in measures
        for issue in measure.issues
    ]
    # 整首的提醒排在前面：没有对应的小节图。
    overall = tempo_text + [f"第 {first} 小节：{where}里没读到，谱里缺了这一小节，请对照{where}补上" if first == last else f"第 {first}~{last} 小节：{where}里没读到，谱里缺了这几小节，请对照{where}补上" for first, last in skipped_measures(measures)]
    (job_dir / "tab_issues.txt").write_text("\n".join(overall + issues) + "\n", encoding="utf-8")
    listed = [{"number": None, "image": None, "issues": [text]} for text in overall]
    listed += [{"number": measure.number, "image": getattr(measure, "image", None), "issues": measure.issues} for measure in measures if measure.issues]
    (job_dir / "tab_issues.json").write_text(json.dumps(listed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest["tab"] = tab_summary(measures)
    manifest["artifacts"]["musicxml"] = relative_to_job(job_dir, path)
    manifest["artifacts"]["tab_issues"] = "tab_issues.txt"
    manifest["status"] = "omr_done"
    manifest.pop("error", None)


def export_score(job_dir: Path, manifest: dict[str, Any], formats: tuple[str, ...] | list[str], title: str | None = None) -> list[Path]:
    """从 score.musicxml 导出 Guitar Pro 文件（score.gp / .gpx / .gp5）和重新排版的 PDF（score_engraved.pdf）。"""
    from video2score.engrave import musicxml_to_pdf
    from video2score.gpfile import export

    source = job_dir / "score.musicxml"
    if not source.is_file():
        raise RuntimeError("还没有 score.musicxml：先识别 TAB（只有五线谱的谱还不能导出）。")
    check_formats(formats)
    artifacts = manifest.setdefault("artifacts", {})
    written = export(source, list(formats), job_dir, "score", title)
    # 边写边记：后面打印 PDF 失败时，已经写好的 Guitar Pro 文件不丢。
    for path in written:
        artifacts[path.suffix[1:]] = relative_to_job(job_dir, path)
    if "pdf" in formats:
        written.append(musicxml_to_pdf(source, job_dir / "score_engraved.pdf"))
        artifacts["engraved_pdf"] = relative_to_job(job_dir, written[-1])
    manifest.pop("export_error", None)
    return written


def _notation(measures: list[CleanMeasure]) -> str:
    kinds = Counter(measure.kind for measure in measures)
    return kinds.most_common(1)[0][0]


# 乐谱开头那一段（乐器名、调弦字母）存在小节库里。
_PREFIX = "measures/prefix.png"


def write_score_pdf(
    path: Path,
    cleaned: list[CleanMeasure],
    numbers: list[int | None],
    title: str,
    prefix: np.ndarray | None = None,
    artist: str = "",
    credit: str = "",
) -> list:
    """排版写 PDF：每行开头放谱号、调号（小节自带的切掉），上方印这一行第一小节的号；
    第一行最前面放乐谱开头的乐器名、调弦字母（prefix，和第一小节同一线距、同样的行）。
    首页写曲名、原唱，每页页脚写出处（credit：视频作者和链接）。"""
    header = extract_header(cleaned)
    opening = None
    if prefix is not None and cleaned:
        opening = Header(image=prefix, spacing=cleaned[0].spacing, staff_top=PAD_ABOVE * cleaned[0].spacing)
    starts = [content_start(measure, header) if header is not None else None for measure in cleaned]
    for measure, start in zip(cleaned, starts):
        if start is not None:
            erase_label_after_header(measure, start)
    rests = [_is_multirest(measure) for measure in cleaned]
    pages = paginate(layout_rows(cleaned, header=header, starts=starts, numbers=fill_numbers(numbers, rests), opening=opening))
    write_pdf(path, pages, title=title, artist=artist, credit=credit)
    return pages


def _is_multirest(measure: CleanMeasure) -> bool:
    """干净图里五线谱中线上有合并休止的粗横杠。"""
    geo = geometry(measure.staff_top, measure.lines, measure.layout, measure.kind, measure.spacing, measure.left_bar, measure.right_bar)
    return has_multirest((255.0 - measure.image.astype(np.float32)) / 255.0, geo)


def fill_numbers(numbers: list[int | None], rests: list[bool] | None = None) -> list[int | None]:
    """读不出的号按前一个号 +1 补上。第一格读不出时算第 1 小节（开头那格带谱号，号常读不出；
    它也常是合并休止，按后面的号往回推会推错）。

    rests 标出哪几格是合并休止：它后面那格不知道跳了几个号，不按 +1 补，改从后面读到的号往回推。
    """
    filled = list(numbers)
    rests = rests or [False] * len(filled)
    if filled and filled[0] is None and any(value is not None and value > 1 for value in filled):
        filled[0] = 1
    for index in range(1, len(filled)):
        if filled[index] is None and filled[index - 1] is not None and not rests[index - 1]:
            filled[index] = filled[index - 1] + 1
    for index in range(len(filled) - 2, -1, -1):
        if filled[index] is None and filled[index + 1] is not None and not rests[index]:
            filled[index] = max(1, filled[index + 1] - 1)
    return filled


def rewrite_pdf(job_dir: Path, link: str | None = None) -> Path:
    """用小节库里已经清洗好的图重新排版（不重跑视频）。"""
    manifest = load_manifest(job_dir)
    if manifest.get("vector"):
        raise RuntimeError("这个作业是矢量 PDF 转的，score.pdf 就是原谱，不用重新排版。")
    video_of(manifest, link)
    title, artist, credit = score_info(manifest)
    library = json.loads((job_dir / "measures.json").read_text(encoding="utf-8"))
    cleaned = [
        CleanMeasure(
            image=read_gray(job_dir / record["image"]),
            spacing=float(record["spacing"]),
            staff_top=float(record["staff_top"]),
            lines=tuple(record["lines"]),
            layout=tuple(tuple(item) for item in record["layout"]),
            kind=record["kind"],
            left_bar=int(record["bars"][0]),
            right_bar=int(record["bars"][1]),
        )
        for record in library
    ]
    path = job_dir / "score.pdf"
    prefix = read_gray(job_dir / _PREFIX) if (job_dir / _PREFIX).is_file() else None
    pages = write_score_pdf(path, cleaned, [record.get("number") for record in library], title, prefix, artist, credit)
    manifest["pages"] = len(pages)
    save_manifest(job_dir, manifest)
    return path


def video_of(manifest: dict[str, Any], link: str | None = None, say: Callable[[str], None] | None = None) -> VideoInfo | None:
    """作业的视频信息。manifest 里存了就直接用；没存（或另给了原视频链接 link）就按来源查一次，存进 manifest。

    本地视频没给链接时没有可查的。查不到不报错（谱照样出，只是没有曲名、出处）。
    """
    source = manifest["source"]
    if link is None and source.get("video"):
        return VideoInfo.from_dict(source["video"])
    try:
        kind, ref = classify_source(link) if link else (source["type"], source["ref"])
        info = lookup(kind, ref)
    except Exception as exc:
        if say is not None:
            say(f"没查到视频信息：{exc}")
        return None
    if info is not None:
        source["video"] = info.to_dict()
    return info


def score_info(manifest: dict[str, Any]) -> tuple[str, str, str]:
    """(曲名, 原唱, 出处)。没有视频信息时曲名按来源（文件夹名、链接），原唱和出处空着。"""
    source = manifest["source"]
    info = VideoInfo.from_dict(source["video"]) if source.get("video") else None
    if info is None:
        return _title(source["type"], source["ref"]), "", ""
    return info.song or info.title or _title(source["type"], source["ref"]), info.artist, info.credit()


def _title(kind: str, ref: str) -> str:
    if kind != "local":
        return ref
    path = Path(ref)
    # 界面上拖进来的视频存在上传目录里：用文件名（不带扩展名）。
    if path.parent == UPLOADS_DIR:
        return path.stem
    # source.mp4、video.mp4 这种通用名没意义，用所在文件夹名（一般是曲名）。
    if path.stem.split("-")[0].lower() in ("source", "video"):
        return path.parent.parent.name if path.parent.name == "download" else path.parent.name
    return path.name


def _save_library(job_dir: Path, slots: list[Slot], cleaned: list[CleanMeasure]) -> Path:
    """每个小节一张干净的图，外加 measures.json：下一步识别 TAB、五线谱时从这里读。"""
    measure_dir = job_dir / "measures"
    measure_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, (slot, measure) in enumerate(zip(slots, cleaned), start=1):
        name = f"{index:04d}.png"
        write_image(measure_dir / name, measure.image)
        raw_name = f"{index:04d}_raw.png"
        if measure.raw is not None:
            write_image(measure_dir / raw_name, measure.raw)
        extra_names = [f"{index:04d}_raw{k}.png" for k in range(1, len(measure.extra_raws) + 1)]
        for extra_name, extra in zip(extra_names, measure.extra_raws):
            write_image(measure_dir / extra_name, extra)
        records.append(
            {
                "index": index,
                "image": f"measures/{name}",
                "raw": f"measures/{raw_name}" if measure.raw is not None else None,
                "raws": [f"measures/{extra_name}" for extra_name in extra_names],
                "kind": measure.kind,
                "layout": [list(item) for item in measure.layout],
                "spacing": round(measure.spacing, 2),
                "staff_top": round(measure.staff_top, 1),
                "lines": [round(line, 1) for line in measure.lines],
                "bars": [measure.left_bar, measure.right_bar],
                "number": slot.number,
                "number_score": round(slot.number_score, 3),
                "seen": slot.seen,
                "first_seconds": round(slot.first_seconds, 2),
            }
        )
    path = job_dir / "measures.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def _write_debug(job_dir: Path, tracker: Tracker, slots: list[Slot]) -> None:
    """帧间对应记录，以及每个小节的小节号区域拼图（用来肉眼检查顺序）。"""
    debug = job_dir / "debug"
    debug.mkdir(exist_ok=True)
    lines = [
        f"{item.seconds:8.2f} {item.kind:6s} offset=({item.offset[0]:.0f},{item.offset[1]:.0f}) "
        f"matched={item.matched} added={item.added}"
        for item in tracker.transitions
    ]
    (debug / "transitions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cells: list[np.ndarray] = []
    for index, slot in enumerate(slots, start=1):
        cell = np.full((48, 110), 255, np.uint8)
        label = next((item.label for item in reversed(slot.observations) if item.label is not None), None)
        if label is not None:
            image = 255 - label
            scaled = cv2.resize(image, (100, max(1, int(image.shape[0] * 100 / image.shape[1]))))[:32]
            cell[14 : 14 + scaled.shape[0], 5 : 5 + scaled.shape[1]] = scaled
        read = "?" if slot.number is None else str(slot.number)
        cv2.putText(cell, f"{index}: {read}", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)
        cells.append(cell)
    # 每个小节号区域另存一张，方便核对和做识别测试用的样本。
    label_dir = debug / "labels"
    label_dir.mkdir(exist_ok=True)
    for index, slot in enumerate(slots, start=1):
        item = next((m for m in reversed(slot.observations) if m.label is not None), None)
        if item is not None:
            write_image(label_dir / f"{index:04d}_sp{item.spacing:.1f}.png", 255 - item.label)
    per_row = 16
    rows = []
    for start in range(0, len(cells), per_row):
        chunk = cells[start : start + per_row]
        chunk += [np.full((48, 110), 255, np.uint8)] * (per_row - len(chunk))
        rows.append(np.hstack(chunk))
    if rows:
        write_image(debug / "labels.png", np.vstack(rows))
