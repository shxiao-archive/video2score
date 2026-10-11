"""检查干净图有没有把音符弄丢、有没有和原帧不是同一个小节。只用作业目录里存下的图，给 bench-pdf 用。

- 丢墨：原帧（最干净的那一帧）里有、另外几帧大多也有的笔画，干净图里几乎没有。
  视频里印的小节号（排版时重印）、两侧小节线附近不算。
- 串格：干净图和原帧的笔画重合度太低，合成时混进了别的小节。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from video2score.tidy import geometry

# 笔画在干净图里覆盖不到这么多，算丢了。
_COVERED = 0.3
# 干净图和原帧笔画的重合度（Dice）低于这个，算串格。
_SAME = 0.6


@dataclass
class InkReport:
    lost: dict[int, int] = field(default_factory=dict)  # 格号 → 丢了几处笔画
    mixed: list[int] = field(default_factory=list)  # 串格的格号

    def summary(self) -> str:
        lost = ",".join(str(index) for index in self.lost) or "无"
        mixed = ",".join(str(index) for index in self.mixed) or "无"
        return f"丢墨 {len(self.lost)} 格：{lost}；串格 {len(self.mixed)} 格：{mixed}"


def check_job(job_dir: Path) -> InkReport:
    report = InkReport()
    for record in json.loads((job_dir / "measures.json").read_text(encoding="utf-8")):
        clean = cv2.imread(str(job_dir / record["image"]), cv2.IMREAD_GRAYSCALE)
        paths = [record.get("raw")] + list(record.get("raws") or [])
        frames = [cv2.imread(str(job_dir / path), cv2.IMREAD_GRAYSCALE) for path in paths if path]
        frames = [frame for frame in frames if frame is not None and clean is not None and frame.shape == clean.shape]
        if clean is None or not frames:
            continue
        lost, dice = check_measure(clean, frames, record)
        if lost:
            report.lost[record["index"]] = lost
        if dice < _SAME:
            report.mixed.append(record["index"])
    return report


def check_measure(clean: np.ndarray, frames: list[np.ndarray], record: dict) -> tuple[int, float]:
    """返回（丢了几处笔画，干净图和原帧的笔画重合度）。frames[0] 是原帧，其余是另外几帧。"""
    spacing = float(record["spacing"])
    bars = record["bars"]
    geo = geometry(record["staff_top"], tuple(record["lines"]), record["layout"], record["kind"], spacing, bars[0], bars[-1])
    mask = _region(clean.shape, geo)
    # 只查够黑的笔画：谱表后面的灰色插画轮廓也是细线，但没有音符那么黑。
    raw = strokes(255.0 - frames[0].astype(np.float32)) & (frames[0] < 105) & mask
    if len(frames) >= 3:
        # 另外几帧大多也有的才算：只在一帧里有的是背景画。
        support = sum(cv2.dilate(strokes(255.0 - frame.astype(np.float32)).astype(np.uint8), np.ones((3, 3), np.uint8)) for frame in frames[1:])
        raw &= support * 2 >= len(frames) - 1
    raw = _without_lines(raw, spacing)
    ink = clean < 128
    near = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
    lost = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < (0.35 * spacing) ** 2:
            continue
        if float(near[labels == index].mean()) < _COVERED:
            lost += 1
    mine = _without_lines(ink & mask, spacing)
    grown_raw = cv2.dilate(raw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    grown_mine = cv2.dilate(mine.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    total = int(raw.sum() + mine.sum())
    dice = float((raw & grown_mine).sum() + (mine & grown_raw).sum()) / total if total else 1.0
    return lost, dice


def strokes(darkness: np.ndarray) -> np.ndarray:
    """细的深色笔画（和 clean._strokes 一样）：黑顶帽去掉大片的深色背景。"""
    image = np.clip(darkness, 0, 255).astype(np.uint8)
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.subtract(image, background) >= 90


def _region(shape: tuple[int, ...], geo) -> np.ndarray:
    """要查的地方：谱表上下 3 个线距以内，去掉两侧小节线附近和视频里的小节号区域。"""
    spacing = geo.spacing
    mask = np.zeros(shape, bool)
    mask[max(0, int(geo.top - 3 * spacing)) : int(geo.bottom + 3 * spacing), int(geo.left + spacing) : max(0, int(geo.right - spacing))] = True
    label_y0, label_y1 = max(0, int(geo.top - 3.0 * spacing)), int(geo.top + 0.3 * spacing)
    mask[label_y0:label_y1, : int(geo.left + 5.5 * spacing)] = False
    mask[label_y0:label_y1, max(0, int(geo.right - 2.5 * spacing)) :] = False
    return mask


def _without_lines(mask: np.ndarray, spacing: float) -> np.ndarray:
    solid = mask.astype(np.uint8)
    lines = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((1, max(5, int(round(1.5 * spacing)))), np.uint8))
    return cv2.subtract(solid, lines) > 0
