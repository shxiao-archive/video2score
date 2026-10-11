"""画测试和短视频用的五线谱画面。识别器不读这个模块。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from video2score.pitches import position_for_midi

CANVAS = (640, 360)
PAGE = (80, 40, 560, 320)
LINE_TOP = 78
LINE_GAP = 22


def draw_measure_frame(pitches: list[int], highlight_x: int) -> Image.Image:
    """深色播放器边框里的一页谱。highlight_x 是页内高亮条的横坐标。"""
    image = Image.new("RGB", CANVAS, (28, 30, 36))
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = PAGE
    draw.rectangle(PAGE, fill=(255, 255, 255))
    band_left = x0 + int(highlight_x)
    draw.rectangle(
        (band_left, y0 + 36, band_left + 34, y1 - 24),
        fill=(255, 214, 0),
    )
    line_ys = [y0 + LINE_TOP + index * LINE_GAP for index in range(5)]
    for y in line_ys:
        draw.line((x0 + 28, y, x1 - 28, y), fill=(0, 0, 0), width=3)
    bottom = line_ys[-1]
    for index, pitch in enumerate(pitches):
        position = position_for_midi(int(pitch))
        cy = bottom - position * (LINE_GAP / 2)
        cx = x0 + 78 + index * 70
        radius = LINE_GAP * 0.42
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(0, 0, 0))
    return image


def draw_scrolling_staff(measures: list[list[int]], highlight_x: int) -> Image.Image:
    """一帧里并排两个小节。后一帧把左侧小节换成新内容，用来检查横滚去重。"""
    width, height = 640, 200
    slot = width // len(measures)
    image = Image.new("RGB", (width, height), (32, 34, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 28, width - 1, height - 24), fill=(238, 238, 238))
    band_left = max(0, min(width - 36, int(highlight_x)))
    draw.rectangle((band_left, 40, band_left + 34, height - 36), fill=(255, 214, 0))
    gap = 16
    line_top = 58
    line_ys = [line_top + index * gap for index in range(5)]
    for y in line_ys:
        draw.line((8, y, width - 9, y), fill=(0, 0, 0), width=3)
    for index in range(len(measures) + 1):
        x = min(width - 2, index * slot)
        draw.line((x, line_ys[0] - 6, x, line_ys[-1] + 6), fill=(0, 0, 0), width=3)
    bottom = line_ys[-1]
    for slot_index, pitches in enumerate(measures):
        for note_index, pitch in enumerate(pitches):
            position = position_for_midi(int(pitch))
            cy = bottom - position * (gap / 2)
            cx = slot_index * slot + 48 + note_index * 56
            radius = gap * 0.42
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(0, 0, 0))
    return image


TAB_GAP = 16
STRIP_MARGIN = 20


def draw_tab_strip(
    measures: list[list[str]],
    *,
    first_number: int = 1,
    measure_width: int = 240,
    top: int = 64,
    height: int = 170,
    lines: int = 4,
    numbers: bool = True,
) -> np.ndarray:
    """一整条横排的 TAB 谱（RGB）。每小节上方左侧印灰色小节号，品位数字压在最下面一条线上。"""
    width = STRIP_MARGIN * 2 + measure_width * len(measures)
    strip = np.full((height, width, 3), 250, np.uint8)
    ys = [top + index * TAB_GAP for index in range(lines)]
    for y in ys:
        cv2.line(strip, (STRIP_MARGIN, y), (width - STRIP_MARGIN, y), (30, 30, 30), 2)
    for index in range(len(measures) + 1):
        x = STRIP_MARGIN + index * measure_width
        cv2.line(strip, (x, ys[0]), (x, ys[-1]), (30, 30, 30), 2)
    for index, frets in enumerate(measures):
        x0 = STRIP_MARGIN + index * measure_width
        if numbers:
            # 字高约 14 像素，和 1080p 视频里的小节号差不多。
            cv2.putText(
                strip, str(first_number + index), (x0 + 4, ys[0] - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (110, 110, 110), 1, cv2.LINE_AA,
            )
        count = max(1, len(frets))
        for position, fret in enumerate(frets):
            cx = x0 + 30 + int(position * (measure_width - 60) / max(1, count - 1)) if count > 1 else x0 + measure_width // 2
            y = ys[-1] - (position % lines) * TAB_GAP
            (text_w, text_h), _base = cv2.getTextSize(fret, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(strip, (cx - text_w // 2 - 2, y - text_h // 2 - 2), (cx + text_w // 2 + 2, y + text_h // 2 + 2), (250, 250, 250), -1)
            cv2.putText(strip, fret, (cx - text_w // 2, y + text_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return strip


def band_frame(strip: np.ndarray, offset: int, *, size: tuple[int, int] = (640, 360), band_y: int = 110) -> np.ndarray:
    """把谱条的一段贴在彩色渐变背景上，模拟横滚谱带。返回 RGB。"""
    width, height = size
    gradient = np.linspace(40, 200, width, dtype=np.float32)
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, :, 0] = gradient.astype(np.uint8)
    frame[:, :, 1] = 90
    frame[:, :, 2] = (240 - gradient).astype(np.uint8)
    window = strip[:, offset : offset + width]
    frame[band_y : band_y + window.shape[0], : window.shape[1]] = window
    return frame


def write_measure_video(
    path: Path,
    measures: list[list[int]],
    *,
    frames_per_measure: int = 8,
    fps: int = 10,
) -> None:
    """每个小节重复若干帧，高亮条逐帧移动，用来检查去重。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        CANVAS,
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法写入视频：{path}")
    try:
        for measure in measures:
            for frame_index in range(frames_per_measure):
                image = draw_measure_frame(measure, highlight_x=18 + frame_index * 14)
                bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
    finally:
        writer.release()
