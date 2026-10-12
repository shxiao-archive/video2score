"""从视频里按固定间隔取帧，并在帧里找出谱面所在的区域。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from video2score.staff import legacy_systems


@dataclass(frozen=True)
class Frame:
    index: int
    seconds: float
    rgb: np.ndarray


def iter_frames(
    video_path: Path,
    *,
    sample_every: float,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> Iterator[Frame]:
    """按间隔读帧。跳过的帧只 grab 不解码成图，省一半时间。"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 10.0
    step = max(1, int(round(sample_every * fps)))
    start_frame = max(0, int(start_sec * fps))
    end_frame = None if end_sec is None else int(end_sec * fps)
    index = 0
    yielded = False
    try:
        while end_frame is None or index < end_frame:
            if index >= start_frame and (index - start_frame) % step == 0:
                ok, bgr = capture.read()
                if not ok:
                    break
                yielded = True
                yield Frame(index, index / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            elif not capture.grab():
                break
            index += 1
    finally:
        capture.release()
    if not yielded:
        raise RuntimeError("没有读到视频帧。")


def inverted(frame: Frame) -> Frame:
    """黑底白字的谱面反转成白底黑字，后面的检测、去色、排版都按浅底深字来。

    反转不改变色差（最亮通道减最暗通道），彩色高亮照样被当作纸。
    """
    return Frame(frame.index, frame.seconds, 255 - frame.rgb)


def video_info(video_path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 10.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    return fps, count


def score_box(
    rgb: np.ndarray,
    ratios: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None),
    page: bool = False,
) -> tuple[int, int, int, int] | None:
    """谱面在帧里的方框 (x0, y0, x1, y1)。给了比例就按比例，否则按谱线找。找不到返回 None。

    page：事先看过是整页视图（survey）。这时只要有包住谱线的白纸就框白纸，不再要求四周是深色界面
    （打谱软件窗口的角上有图标、按钮，不一定暗）。
    """
    height, width = rgb.shape[:2]
    if any(value is not None for value in ratios):
        x0, y0, x1, y1 = ratios
        return (
            int(round(width * (0.0 if x0 is None else x0))),
            int(round(height * (0.0 if y0 is None else y0))),
            int(round(width * (1.0 if x1 is None else x1))),
            int(round(height * (1.0 if y1 is None else y1))),
        )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    systems = legacy_systems(gray)
    if not systems:
        return None
    page_box = _white_page_box(gray, systems, chrome=not page)
    box = page_box if page_box is not None else _band_box(gray, _best_cluster(systems))
    if box[2] - box[0] < 20 or box[3] - box[1] < 20:
        return None
    return box


def page_box(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """包住谱线、左右不贯通画面的白纸（打谱软件的整页视图）。没有返回 None。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    systems = legacy_systems(gray)
    return _white_page_box(gray, systems, chrome=False) if systems else None


def crop_score_page(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """按等距谱线裁出谱带。谱带若落在深色界面里的白纸上，则裁整页。"""
    box = score_box(np.array(image.convert("RGB")))
    if box is None:
        raise RuntimeError("没有找到谱面。")
    return image.crop(box), box


def _best_cluster(
    systems: list[tuple[float, float, float, int]],
) -> list[tuple[float, float, float, int]]:
    """把挨在一起的谱表收成一条谱带，取线数最多的那条。"""
    ordered = sorted(systems, key=lambda item: item[0])
    clusters: list[list[tuple[float, float, float, int]]] = [[ordered[0]]]
    for system in ordered[1:]:
        top, _bottom, _spacing, _count = system
        prev_top, prev_bottom, prev_spacing, _prev_count = clusters[-1][-1]
        gap = top - prev_bottom
        height = max(1.0, prev_bottom - prev_top)
        limit = max(height * 1.8, prev_spacing * 7.0)
        if _prev_count == 5 and system[2] >= 1.15 * prev_spacing:
            # 五线谱下面线距更大的一组是它的 TAB：有的横向滚动谱把 TAB 放得很远（9~11 个线距），
            # 和 staff._system_groups 配对的上限一样放宽，不然谱面框只框住五线谱。
            limit = max(height * 2.2, prev_spacing * 12.0)
        if gap <= limit:
            clusters[-1].append(system)
        else:
            clusters.append([system])

    def rank(cluster: list[tuple[float, float, float, int]]) -> tuple[int, float]:
        lines = sum(item[3] for item in cluster)
        span = cluster[-1][1] - cluster[0][0]
        return (lines, span)

    return max(clusters, key=rank)


def _band_box(
    gray: np.ndarray,
    cluster: list[tuple[float, float, float, int]],
) -> tuple[int, int, int, int]:
    """上方多留一截，把加线、小节号和演奏记号包进来。走到深色画面就停。"""
    height, width = gray.shape
    top = min(item[0] for item in cluster)
    bottom = max(item[1] for item in cluster)
    spacing = cluster[0][2]
    top_pad = max(8, int(round(spacing * 2.0)))
    extra = max(top_pad, int(round(spacing * 7.0)))
    bottom_pad = max(8, int(round(cluster[-1][2] * 1.15)))
    y0 = max(0, int(top - top_pad))
    limit = max(0, int(top - extra))
    while y0 > limit and float(np.median(gray[y0 - 1])) >= 190:
        y0 -= 1
    y1 = min(height, int(bottom + bottom_pad) + 1)
    return (0, y0, width, y1)


def _white_page_box(
    gray: np.ndarray,
    systems: list[tuple[float, float, float, int]],
    chrome: bool = True,
) -> tuple[int, int, int, int] | None:
    """深色界面里、包住谱线的白纸。满幅亮底或小亮斑不算。

    chrome=False（整页视图）：不查四周是不是深色界面；白纸可以碰到画面上下沿（翻页、滚动时一张纸只露出一截）、
    一侧粘上界面上浅色的滚动条（框的左右两边不够白的列由 _trim_white_page 去掉），
    上下叠着露出两张纸时框住两张（中间的灰缝由 measures._band_edges 挡在小节外面）。
    """
    if chrome and not _dark_chrome(gray):
        return None
    height, width = gray.shape
    mask = np.where(gray >= 200, np.uint8(255), np.uint8(0))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    found = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    best: tuple[int, int, int, int] | None = None
    best_count = 0
    best_area = 0
    pages: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_w, box_h = (int(value) for value in cv2.boundingRect(contour))
        if box_w < width * 0.4 or box_h < 20:
            continue
        if chrome and (x <= 8 or y <= 8 or x + box_w >= width - 8 or y + box_h >= height - 8):
            continue
        # 整页视图：纸的一侧可能和界面上浅色的滚动条连成一片（修边以后再查左右边沿），但不会左右都贯通（那是横贯画面的谱带）。
        if not chrome and x <= 8 and x + box_w >= width - 8:
            continue
        count = sum(
            1
            for top, bottom, _spacing, _count in systems
            if y <= (top + bottom) / 2.0 <= y + box_h and x <= width / 2 <= x + box_w
        )
        area = box_w * box_h
        if count:
            pages.append((x, y, x + box_w, y + box_h))
        if count > best_count or (count == best_count and count and area > best_area):
            best = (x, y, x + box_w, y + box_h)
            best_count = count
            best_area = area
    if best_count == 0 or best is None:
        return None
    if not chrome:
        stacked = [box for box in pages if min(box[2], best[2]) - max(box[0], best[0]) >= 0.8 * (best[2] - best[0])]
        best = (min(box[0] for box in stacked), min(box[1] for box in stacked), max(box[2] for box in stacked), max(box[3] for box in stacked))
    trimmed = _trim_white_page(gray, best, systems)
    if not chrome and (trimmed[0] <= 8 or trimmed[2] >= width - 8):
        # 修掉两边不够白的列以后还碰着画面左右沿：是贴着画面边上的白底（谱带旁边的白色竖栏），不是软件界面里的一张纸。
        return None
    return trimmed


def _trim_white_page(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    systems: list[tuple[float, float, float, int]],
) -> tuple[int, int, int, int]:
    """白纸框的外沿会粘上标题栏和时间轴，把两端不够白的行去掉；左右粘上的水印、按钮，把两边不够白的列去掉。"""
    x0, y0, x1, y1 = box
    if y1 - y0 < 16:
        return box
    bright = (gray[y0:y1, x0:x1] >= 200).mean(axis=1)
    mids = [
        (top + bottom) / 2.0
        for top, bottom, _spacing, _count in systems
        if y0 <= (top + bottom) / 2.0 <= y1
    ]
    if not mids:
        return box
    first = int(min(mids) - y0)
    last = int(max(mids) - y0)
    top = 0
    while top < first and float(bright[top]) < 0.55:
        top += 1
    bottom = len(bright)
    while bottom - 1 > last and float(bright[bottom - 1]) < 0.75:
        bottom -= 1
    if bottom - top < 20:
        return box
    columns = (gray[y0 + top : y0 + bottom, x0:x1] >= 200).mean(axis=0)
    left, right = 0, len(columns)
    while left < right and float(columns[left]) < 0.5:
        left += 1
    while right > left and float(columns[right - 1]) < 0.5:
        right -= 1
    if right - left < 20:
        return (x0, y0 + top, x1, y0 + bottom)
    return (x0 + left, y0 + top, x0 + right, y0 + bottom)


def _dark_chrome(gray: np.ndarray) -> bool:
    height, width = gray.shape
    if height < 16 or width < 16:
        return False
    patches = (
        gray[:8, :8],
        gray[:8, width - 8 :],
        gray[height - 8 :, :8],
        gray[height - 8 :, width - 8 :],
    )
    return all(float(patch.mean()) < 90 for patch in patches)
