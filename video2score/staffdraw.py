"""画测试和自检用的 TAB 谱画面。识别器不读这个模块。"""

from __future__ import annotations

import cv2
import numpy as np

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

