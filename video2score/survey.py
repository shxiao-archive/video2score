"""正式逐帧找小节之前，先在整段视频里均匀抽一些帧，看谱面是哪一种，后面按对应的模式处理。

两种谱面：
- 谱带（band）：横贯画面的一行谱，横着滚动或一格一格跳。大多数动态谱视频是这种。
- 整页（page）：打谱软件的页面视图录屏，一屏是一张白纸上好几行谱，上下滚动或翻页。
  按白纸认：包住谱线、左右两边是软件界面（不贯通画面）的白纸。不按每帧几行谱认，谱带的框有时也框进两三行。

另看要不要放大：谱面小时数字、滑音线、连线挤在两三个像素里，各处按线距定的门槛都不准，放大到线距约 _TARGET 像素
再找小节常常读得准得多；但同样大小的谱面，有的放大后反而读得差（原图本来清楚，放大只是糊开）。光看线距定不了，
所以实测：抽几帧，原尺寸和放大后各读一遍 TAB 品位和小节号，放大后有把握的读数明显多才放大。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from video2score.frames import Frame, inverted, page_box, score_box
from video2score.measures import MeasureFinder
from video2score.ocr import read_measure
from video2score.staff import find_systems
from video2score.tab import read_frets, tab_staff

# 抽多少帧看谱面类型和线距。
_SAMPLES = 24
# 其中多少帧拿来试放大。
_TRIALS = 8
# 放大到这个线距。测试视频里 1080p 的谱带线距是 15 像素左右，各处门槛按这个尺寸调过。
_TARGET = 15.0
# 放大不到这么多倍的不试（谱面已经够大）。
_MIN_SCALE = 1.05
# 放大后有把握的读数（品位、小节号）要比原尺寸多这么多倍才放大：放大以后要处理的像素多好几倍，慢得多。
_GAIN = 1.2
# 读数有这么有把握才算数。
_SURE = 0.8


@dataclass(frozen=True)
class Mode:
    layout: str = "band"
    # 视频里原本的线距（像素）；没抽到谱面是 0。
    spacing: float = 0.0
    # 读帧以后放大的倍数，1 是不放大。
    scale: float = 1.0
    # 抽到的帧里每帧几行谱（中位数）。
    rows: float = 1.0
    # 试过的放大倍数和各自有把握的读数个数，如 ((1.0, 66), (1.94, 138))。谱面够大、没试时是空的。
    trials: tuple[tuple[float, int], ...] = ()

    def describe(self) -> str:
        kind = "整页视图" if self.layout == "page" else "谱带"
        text = f"{kind}，线距 {self.spacing:.1f} 像素"
        if len(self.trials) == 2:
            (_one, plain), (larger, enlarged) = self.trials
            verdict = f"放大 {self.scale:.2f} 倍识别" if self.scale > 1 else "按原尺寸识别"
            return text + f"；试读有把握的：原尺寸 {plain} 个，放大 {larger:.2f} 倍 {enlarged} 个，{verdict}"
        return text

    def to_json(self) -> dict:
        def tidy(value):
            if isinstance(value, float):
                return round(value, 3)
            if isinstance(value, (tuple, list)):
                return [tidy(item) for item in value]
            return value

        return {key: tidy(value) for key, value in asdict(self).items()}


def survey(
    video_path,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    ratios: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None),
) -> Mode:
    """在 start_sec~end_sec 里均匀抽 _SAMPLES 帧：一半以上是整页视图的白纸就按整页处理；线距取所有谱行的中位数。
    谱面比 _TARGET 小时，从中再挑 _TRIALS 帧试读，定要不要放大。"""
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 10.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        first = int(start_sec * fps)
        last = count if end_sec is None else min(count, int(end_sec * fps))
        rows: list[int] = []
        pages = 0
        spacings: list[float] = []
        seen: list[Frame] = []
        for k in range(_SAMPLES):
            index = first + int((last - first) * (k + 0.5) / _SAMPLES)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok:
                continue
            frame = Frame(index, index / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            found, page = _systems(frame, ratios)
            if found:
                rows.append(len(found))
                pages += page
                spacings.extend(system.spacing for system in found)
                seen.append(frame)
    finally:
        capture.release()
    if not rows:
        return Mode()
    spacing = float(np.median(spacings))
    layout = "page" if pages * 2 > len(rows) else "band"
    scale = 1.0
    trials: tuple[tuple[float, int], ...] = ()
    larger = _TARGET / spacing
    if larger >= _MIN_SCALE:
        picked = [seen[int(len(seen) * (k + 0.5) / _TRIALS)] for k in range(min(_TRIALS, len(seen)))]
        plain = _confident(picked, ratios, 1.0, layout == "page")
        enlarged = _confident(picked, ratios, larger, layout == "page")
        trials = ((1.0, plain), (larger, enlarged))
        if enlarged > _GAIN * plain:
            scale = larger
    return Mode(layout=layout, spacing=spacing, scale=scale, rows=float(np.median(rows)), trials=trials)


def _systems(frame: Frame, ratios) -> tuple[list, bool]:
    """这一帧的谱行，以及是不是整页视图的白纸。手动给了谱面区域（ratios）就只按它框。"""
    manual = any(value is not None for value in ratios)
    for candidate in (frame, inverted(frame)):
        page = None if manual else page_box(candidate.rgb)
        box = page or score_box(candidate.rgb, ratios)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        found = find_systems(np.ascontiguousarray(candidate.rgb[y0:y1, x0:x1]))
        if found:
            return found, page is not None
    return [], False


def _confident(frames: list[Frame], ratios, scale: float, page: bool) -> int:
    """按这个放大倍数切小节，用通用模板读 TAB 品位和小节号，数有把握的读数。"""
    finder = MeasureFinder(ratios, scale, page)
    total = 0
    for frame in frames:
        found = finder(frame)
        if found is None:
            continue
        for measure in found.measures:
            ox, oy = measure.origin
            staff = tab_staff({"staff_top": measure.top - oy, "lines": list(measure.lines), "layout": measure.layout})
            # 下沿被画面切掉的小节（darkness 不够高）不读品位。
            if staff is not None and staff.bottom + 0.8 * staff.spacing < measure.darkness.shape[0]:
                image = (255 - measure.darkness).astype(np.uint8)
                total += sum(note.score >= _SURE for note in read_frets(image, staff, bars=(measure.left - ox, measure.right - ox)))
            reading = read_measure(measure)
            total += reading is not None and reading.score >= _SURE
    return total
