"""在去色后的五线谱上，按从左到右的顺序读出符头音高。"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

from video2score.pitches import midi_for_position


def recognize_staff(image: Image.Image) -> list[int]:
    """返回 MIDI 音高，时间顺序为从左到右。"""
    gray = np.array(ImageOps.grayscale(image))
    ink = np.where(gray < 128, np.uint8(255), np.uint8(0))
    line_centers = _staff_line_centers(ink)
    if len(line_centers) < 5:
        raise RuntimeError(f"没有找到五条谱线，只看到 {len(line_centers)} 条。")
    line_centers = sorted(line_centers[:5])
    spacing = (line_centers[-1] - line_centers[0]) / 4.0
    if spacing < 4:
        raise RuntimeError("谱线间距过小。")

    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(spacing * 2), 15), 1))
    lines_only = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horizontal)
    notes = cv2.subtract(ink, lines_only)
    # 压在谱线上的符头会被减去一条，竖向闭合把上下两半接回去。
    bridge = max(3, int(round(spacing * 0.3)))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge))
    notes = cv2.morphologyEx(notes, cv2.MORPH_CLOSE, vertical)
    notes = cv2.morphologyEx(notes, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(notes, connectivity=8)

    found: list[tuple[float, float]] = []
    min_side = spacing * 0.28
    max_side = spacing * 1.35
    for label in range(1, count):
        _x, _y, width, height, area = (int(v) for v in stats[label])
        if area < 20:
            continue
        if width < min_side or height < min_side:
            continue
        if width > max_side or height > max_side * 1.4:
            continue
        if width > height * 3 or height > width * 3:
            continue
        cx, cy = centroids[label]
        found.append((float(cx), float(cy)))

    found = _merge_split_heads(found, spacing)
    pitches: list[int] = []
    half = spacing / 2.0
    bottom = line_centers[-1]
    for _cx, cy in found:
        position = int(round((bottom - cy) / half))
        try:
            pitches.append(midi_for_position(position))
        except ValueError:
            continue
    return pitches


def ink_line_count(image: Image.Image) -> int:
    """去色图上横贯的墨线有几条。多出四线谱时不会是 5。"""
    gray = np.array(ImageOps.grayscale(image))
    ink = np.where(gray < 128, np.uint8(255), np.uint8(0))
    return len(_all_line_centers(ink))


def _all_line_centers(ink: np.ndarray) -> list[float]:
    _height, width = ink.shape
    if width == 0:
        return []
    row_ink = ink.mean(axis=1) / 255.0
    dark_rows = [index for index, frac in enumerate(row_ink) if frac > 0.45]
    if not dark_rows:
        return []
    clusters: list[list[int]] = [[dark_rows[0]]]
    for row in dark_rows[1:]:
        if row - clusters[-1][-1] <= 3:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return sorted(sum(rows) / len(rows) for rows in clusters)


def _staff_line_centers(ink: np.ndarray) -> list[float]:
    centers = _all_line_centers(ink)
    if len(centers) <= 5:
        return centers
    best: list[float] | None = None
    best_score: tuple[float, float] | None = None
    for start in range(0, len(centers) - 4):
        group = centers[start : start + 5]
        gaps = [group[index + 1] - group[index] for index in range(4)]
        mean = sum(gaps) / 4.0
        if mean < 4:
            continue
        variance = sum((gap - mean) ** 2 for gap in gaps) / 4.0
        score = (variance, -mean)
        if best_score is None or score < best_score:
            best_score = score
            best = group
    return best if best is not None else centers[:5]


def _merge_split_heads(
    found: list[tuple[float, float]], spacing: float
) -> list[tuple[float, float]]:
    """同一符头被谱线切开时，中心会靠在一起，合并成一个音。"""
    merged: list[tuple[float, float]] = []
    for cx, cy in sorted(found, key=lambda item: (item[0], item[1])):
        if merged:
            px, py = merged[-1]
            if abs(cx - px) <= spacing * 0.55 and abs(cy - py) <= spacing * 0.75:
                merged[-1] = ((px + cx) / 2.0, (py + cy) / 2.0)
                continue
        merged.append((cx, cy))
    return merged
