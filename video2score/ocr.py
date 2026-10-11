"""读小节号：在小节线上方那一块里切出一串数字，逐字和模板比。

模板有三类：numbers.py 里从真实视频取的手工模板、本机字体渲染的 0~9，
以及每段视频第一遍读出来的可信字形平均出的专属模板（adapt）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from video2score.measures import HEAD_LABEL_LEFT, LABEL_LEFT, Measure
from video2score.numbers import _DIGIT_BANKS, _fit_glyph, _glyph_corr, corr_all

# 本机字体里常见的谱面字体。找不到的跳过。
_FONT_DIRS = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
)
_FONT_FILES = (
    "times.ttf", "timesbd.ttf", "georgia.ttf", "cambria.ttc", "arial.ttf", "arialbd.ttf",
    "segoeui.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf",
    "DejaVuSerif.ttf", "DejaVuSans.ttf", "LiberationSerif-Regular.ttf", "LiberationSans-Regular.ttf",
)
# 认一个字至少要这么像，并且比第二名高出这么多。
MIN_SCORE = 0.55
MIN_MARGIN = 0.04


@dataclass
class Reading:
    number: int
    score: float
    canvases: list[np.ndarray] = field(repr=False, default_factory=list)


Bank = tuple[tuple[int, np.ndarray], ...]


@lru_cache(maxsize=1)
def generic_bank() -> Bank:
    """手工模板加字体渲染的模板：用冻结的（resources/glyphs.npz），没有才用本机字体现画。"""
    from video2score.glyphs import frozen_bank

    return frozen_bank("generic", numeric=True) or _generic_bank_from_fonts()


def _generic_bank_from_fonts() -> Bank:
    banks: list[tuple[int, np.ndarray]] = list(_DIGIT_BANKS)
    for path in _font_paths():
        banks.extend(_render_font(path))
    return tuple(banks)


def _font_paths() -> list[Path]:
    found: list[Path] = []
    for name in _FONT_FILES:
        for directory in _FONT_DIRS:
            if not directory.is_dir():
                continue
            direct = directory / name
            # Linux 的字体分散在子目录里，要往下找。
            hit = direct if direct.is_file() else next(directory.rglob(name), None)
            if hit is not None:
                found.append(hit)
                break
    return found


def _render_font(path: Path) -> list[tuple[int, np.ndarray]]:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(str(path), 64)
    except OSError:
        return []
    rendered: list[tuple[int, np.ndarray]] = []
    for digit in range(10):
        canvas = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(canvas).text((16, 8), str(digit), fill=255, font=font)
        ink = np.array(canvas) > 127
        rows = np.flatnonzero(ink.any(axis=1))
        cols = np.flatnonzero(ink.any(axis=0))
        if rows.size == 0:
            continue
        rendered.append((digit, _fit_glyph(ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1])))
    return rendered


def stroke_mask(label: np.ndarray, valid: np.ndarray | None, spacing: float) -> np.ndarray:
    """比周围暗一截的细笔画。阈值跟着这一块的对比度走：GP 的小节号是很浅的灰。"""
    size = max(7, int(round(spacing * 0.6)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    strokes = cv2.subtract(label, cv2.morphologyEx(label, cv2.MORPH_OPEN, kernel))
    usable = strokes[valid] if valid is not None and valid.any() else strokes.ravel()
    peak = float(np.percentile(usable, 99.7)) if usable.size else 0.0
    mask = strokes >= max(28.0, peak * 0.42)
    if valid is not None:
        mask &= valid
    return mask


def read_label(
    label: np.ndarray | None,
    valid: np.ndarray | None,
    spacing: float,
    bank: Bank | None = None,
    bar_x: float | None = None,
) -> Reading | None:
    """读一块小节号区域。读不出、或不像数字时返回 None。bar_x 是号该挨着的位置（默认是小节线）。"""
    if label is None:
        return None
    bank = bank or generic_bank()
    mask = stroke_mask(label, valid, spacing)
    seen_columns = np.flatnonzero(valid.any(axis=0)) if valid is not None else np.arange(label.shape[1])
    if seen_columns.size == 0:
        return None
    edges = (int(seen_columns[0]), int(seen_columns[-1]))
    found = _pick_blob(mask, spacing, bar_x=LABEL_LEFT * spacing if bar_x is None else bar_x, edges=edges)
    if found is None:
        return None
    blob, bx, by = found
    boxes = _split_digits(blob)
    if not 1 <= len(boxes) <= 3:
        return None
    digits: list[int] = []
    scores: list[float] = []
    canvases: list[np.ndarray] = []
    for y0, y1, x0, x1 in boxes:
        # 切字用笔画，认字用灰度：小字的细斜笔二值化后常断，灰度里还在。
        canvas = fit_gray(label[by + y0 : by + y1, bx + x0 : bx + x1])
        digit, score, margin = classify(canvas, bank)
        if digit is None or score < MIN_SCORE or margin < MIN_MARGIN:
            return None
        digits.append(digit)
        scores.append(score)
        canvases.append(canvas)
    if digits[0] == 0:
        return None
    number = int("".join(str(digit) for digit in digits))
    return Reading(number=number, score=min(scores), canvases=canvases)


def fit_gray(patch: np.ndarray) -> np.ndarray:
    """灰度字缩到 20×12：高固定 18 像素，左右居中，最黑处拉到 1。和 numbers._fit_glyph 同一种摆法。"""
    height, width = patch.shape
    values = patch.astype(np.float32)
    peak = float(values.max())
    if peak <= 0 or height < 1 or width < 1:
        return np.zeros((20, 12), np.float32)
    values /= peak
    resized_w = max(1, min(12, int(round(width * 18.0 / height))))
    small = cv2.resize(values, (resized_w, 18), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((20, 12), np.float32)
    left = (12 - resized_w) // 2
    canvas[1:19, left : left + resized_w] = small
    return canvas


def _pick_blob(
    mask: np.ndarray,
    spacing: float,
    bar_x: float,
    edges: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int, int] | None:
    """整个号码当一块找：笔画膨胀后连成的块，高度像一行字、宽度不超过三位数。

    取离小节线最近、最贴近谱线的那块，返回 (它的笔画, 左上角 x, y)。
    """
    # 横向按字距膨胀，同一个号的几位连成一块；纵向只补断笔。
    reach = max(5, int(round(0.45 * spacing)) | 1)
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((3, reach), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(grown, connectivity=8)
    best: tuple[float, tuple[np.ndarray, int, int]] | None = None
    bottom = mask.shape[0]
    for index in range(1, count):
        x, y, width, height, _area = (int(value) for value in stats[index])
        # 膨胀让高多了两像素、宽多了 reach-1。
        if not 0.45 * spacing <= height - 2 <= 1.45 * spacing:
            continue
        if width - reach > 3.3 * spacing:
            continue
        strokes = mask[y : y + height, x : x + width] & (labels[y : y + height, x : x + width] == index)
        if int(strokes.sum()) < 8:
            continue
        # 贴着左右边缘（或画面截断处）的号可能被截掉了一位，半个 2 会被认成 1。宁可不读。
        first, last = edges if edges is not None else (0, mask.shape[1] - 1)
        columns = np.flatnonzero(strokes.any(axis=0))
        if x + columns[0] <= first or x + columns[-1] >= last:
            continue
        horizontal = 0.0 if x <= bar_x <= x + width else min(abs(x - bar_x), abs(x + width - bar_x))
        cost = horizontal + 0.5 * (bottom - (y + height))
        if best is None or cost < best[0]:
            best = (cost, (strokes, x, y))
    return None if best is None else best[1]


def _split_digits(blob: np.ndarray) -> list[tuple[int, int, int, int]]:
    """按列投影在空白处切开；粘连的两位按字宽在最窄处再切。

    返回每一位的 (y0, y1, x0, x1)，按这一位自己的笔画裁紧。
    """
    columns = blob.any(axis=0)
    pieces: list[tuple[int, int]] = []
    start: int | None = None
    for index, filled in enumerate(list(columns) + [False]):
        if filled and start is None:
            start = index
        elif not filled and start is not None:
            pieces.append((start, index))
            start = None
    filled_rows = np.flatnonzero(blob.any(axis=1))
    height = int(filled_rows[-1] - filled_rows[0]) + 1 if filled_rows.size else 1
    digit_width = max(3.0, 0.62 * height)
    split: list[tuple[int, int]] = []
    for left, right in pieces:
        if int(blob[:, left:right].sum()) < 4:
            continue
        parts = max(1, int(round((right - left) / digit_width)))
        if parts == 1 or (right - left) < 1.3 * digit_width:
            split.append((left, right))
            continue
        profile = blob[:, left:right].sum(axis=0)
        cuts = [left]
        for part in range(1, parts):
            centre = int(round((right - left) * part / parts))
            lo, hi = max(1, centre - 2), min(right - left - 1, centre + 3)
            cuts.append(left + lo + int(np.argmin(profile[lo:hi])))
        cuts.append(right)
        split.extend(zip(cuts, cuts[1:]))
    boxes: list[tuple[int, int, int, int]] = []
    for left, right in split:
        piece = blob[:, left:right]
        rows = np.flatnonzero(piece.any(axis=1))
        cols = np.flatnonzero(piece.any(axis=0))
        if rows.size == 0:
            continue
        # 切下来的碎渣（小于字高一半）不当一位。
        if rows[-1] - rows[0] + 1 < 0.5 * height:
            continue
        boxes.append((int(rows[0]), int(rows[-1]) + 1, left + int(cols[0]), left + int(cols[-1]) + 1))
    return boxes


def classify(canvas: np.ndarray, bank: Bank) -> tuple[int | None, float, float]:
    """(数字, 最像的相关值, 和第二个数字的差距)。

    整页谱每帧十几个小节、每个号要和一百多个模板比：用 corr_all 一次算完。
    """
    scores: dict[int, float] = {}
    for (digit, _template), score in zip(bank, corr_all(canvas, bank)):
        if score > scores.get(digit, -1.0):
            scores[digit] = score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, 0.0, 0.0
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    return ranked[0][0], ranked[0][1], ranked[0][1] - second


def read_measure(measure: Measure, bank: Bank | None = None) -> Reading | None:
    """读一个小节的号。谱行第一小节的号印在谱号、调号、拍号后面：小节线上方读不出时再到那儿读。"""
    reading = read_label(measure.label, measure.label_valid, measure.spacing, bank)
    if reading is not None or measure.head_label is None:
        return reading
    return read_label(measure.head_label, measure.head_label_valid, measure.spacing, bank, bar_x=HEAD_LABEL_LEFT * measure.spacing)


def adapted_bank(samples: dict[int, list[np.ndarray]], minimum: int = 3) -> Bank:
    """这段视频里可信读数的字形按数字取平均，放在通用模板前面。"""
    own: list[tuple[int, np.ndarray]] = []
    for digit, canvases in samples.items():
        if len(canvases) >= minimum:
            own.append((digit, np.mean(np.stack(canvases), axis=0).astype(np.float32)))
    return tuple(own) + generic_bank()
