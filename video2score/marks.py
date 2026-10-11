"""谱面上的几样记号：TAB 左边的调弦字母、第一小节上方的速度。都在清洗后的图上读（白底黑字）。

- 调弦：乐谱开头截到的那一段（measures/prefix.png）里，每条 TAB 线的高度上最靠右的那个字母（左边竖排的是乐器名）。
  字母按本机字体渲染的模板认，只在离标准调弦两个半音以内挑（Drop D 的最低弦是 D1）。
- 速度：第一小节上方"♩ = 140"：找到等号（上下两道短横），读它右边那几位数字。
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from video2score.numbers import _fit_glyph, _glyph_corr
from video2score.tab import _font_paths, _render, tab_bank

_LETTERS = "CDEFGAB"
_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# 不是标准调弦的字母，要比标准的那个像这么多才换。
_SWITCH = 0.1


@lru_cache(maxsize=1)
def _letter_bank() -> tuple[tuple[str, np.ndarray], ...]:
    """调弦字母：用冻结的（resources/glyphs.npz），没有才用本机字体现画。"""
    from video2score.glyphs import frozen_bank

    return frozen_bank("letters") or _letter_bank_from_fonts()


def _letter_bank_from_fonts() -> tuple[tuple[str, np.ndarray], ...]:
    bank: list[tuple[str, np.ndarray]] = []
    for path in _font_paths():
        bank.extend(_render(path, _LETTERS))
    return tuple(bank)


def read_tuning(prefix: np.ndarray, tab_lines: list[float], standard: tuple[int, ...]) -> tuple[int, ...] | None:
    """每根弦的空弦音高（MIDI），第 1 弦（最上面那条线）在前。tab_lines 是 TAB 各线在 prefix 图里的 y。

    有一根读不出，整个不用，返回 None。
    """
    if len(tab_lines) != len(standard):
        return None
    ink = prefix < 128
    gap = (tab_lines[-1] - tab_lines[0]) / max(1, len(tab_lines) - 1)
    tuning: list[int] = []
    for y, open_string in zip(tab_lines, standard):
        found = _letter_at(ink, y, gap)
        if found is None:
            return None
        crop, alter = found
        canvas = _fit_glyph(crop)
        scores: dict[str, float] = {}
        for label, template in _letter_bank():
            scores[label] = max(scores.get(label, -1.0), _glyph_corr(canvas, template))
        # 十来个像素高的字，E 和 B、D 的相关值只差一点：只在离标准调弦两个半音以内的音里挑（降调、Drop D），
        # 标准的那个优先，别的要明显更像才换。
        options = {}
        for label, score in scores.items():
            pitch = min((12 * octave + _SEMITONES[label] + alter for octave in range(0, 9)), key=lambda value: abs(value - open_string))
            if abs(pitch - open_string) <= 2:
                options[pitch] = max(options.get(pitch, -1.0), score)
        # 带升降号时（整体降半音），标准调弦的字母降半音后是首选。
        expected = open_string + alter
        if expected not in options or options[expected] < 0.4:
            return None
        pitch = max(options, key=lambda value: options[value] - (0.0 if value == expected else _SWITCH))
        # D 和 E（Drop D 就差在这根弦上）按宽窄分：D 的圆肚让它和高差不多宽，E 窄得多。
        names = {value % 12 for value in (pitch, open_string)}
        if not alter and names == {_SEMITONES["D"], _SEMITONES["E"]}:
            wide = crop.shape[1] >= 0.95 * crop.shape[0]
            pitch = next(value for value in (pitch, open_string) if (value % 12 == _SEMITONES["D"]) == wide)
        tuning.append(pitch)
    return tuple(tuning)


def _letter_at(ink: np.ndarray, y: float, gap: float) -> tuple[np.ndarray, int] | None:
    """TAB 线 y 上、最靠右的那个字母（左边竖排的是乐器名），以及它后面的升降号（降半音调弦写成 "G♭"）：-1、0、+1。
    按列分段时不看线所在的那两三行（伸进来的谱线段）。"""
    top, bottom = max(0, int(y - 0.45 * gap)), min(ink.shape[0], int(y + 0.45 * gap) + 1)
    window = ink[top:bottom]
    if window.size == 0:
        return None
    rows = np.ones(window.shape[0], bool)
    line = int(round(y)) - top
    rows[max(0, line - 1) : line + 2] = False
    columns = np.flatnonzero(window[rows].any(axis=0))
    if columns.size == 0:
        return None
    # 按列分段（中间空一两列也算同一个字），从右往左。
    pieces: list[tuple[int, int]] = []
    for column in columns[::-1]:
        column = int(column)
        if pieces and pieces[-1][0] - column <= 3:
            pieces[-1] = (column, pieces[-1][1])
        else:
            pieces.append((column, column))
    alter = 0
    start, end = pieces[0]
    # 升降号的竖笔比字母高、伸到窗口最上面两行，字母碰不到：从这根竖笔切开（字母的衬线可能和它连成一段）。
    tall = [column for column in range(start, end + 1) if window[:2, column].any()]
    if tall and tall[0] > start + 0.4 * (end - start):
        sign = _accidental(window[:, tall[0] : end + 1], tall[0] - start)
        if sign:
            alter = sign
            end = tall[0] - 1
            while end > start and not window[rows, end].any():
                end -= 1
    elif tall and len(pieces) >= 2:
        sign = _accidental(window[:, start : end + 1], pieces[1][1] - pieces[1][0] + 1)
        if sign:
            alter = sign
            start, end = pieces[1]
    piece = window[:, start : end + 1]
    used = np.flatnonzero(piece.any(axis=1))
    if used.size == 0 or used[-1] - used[0] < 0.3 * gap:
        return None
    return piece[used[0] : used[-1] + 1], alter


def _accidental(piece: np.ndarray, letter_width: int) -> int:
    """字母右边的一块是不是升降号：比字母窄得多、竖笔从窗口顶上伸下来（字母不会碰到窗口上沿）。
    调弦几乎都是往下调，按降号（-1）算；只有两道几乎贯穿整个窗口的竖笔才是升号（+1）。不是升降号返回 0。"""
    height, width = piece.shape
    if width > 0.6 * letter_width or not piece[:2].any():
        return 0
    strokes = np.flatnonzero(piece.sum(axis=0) >= 0.8 * height)
    groups = 1 + int(np.count_nonzero(np.diff(strokes) > 1)) if strokes.size else 0
    return 1 if groups >= 2 else -1


def read_tempo(image: np.ndarray, staff_top: float, spacing: float, bars: tuple[int, int]) -> int | None:
    """第一小节上方的"♩ = 数字"。找等号：两道一样长的短横上下叠着；读它右边、同一高度的几位数字。"""
    y0 = max(0, int(staff_top - 4.5 * spacing))
    y1 = max(y0, int(staff_top - 0.3 * spacing))
    region = (image[y0:y1, bars[0] : bars[1]] < 128).astype(np.uint8)
    if region.size == 0:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
    bars_found = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h, _area in stats[1:]
        if 0.4 * spacing <= w <= 1.4 * spacing and h <= 0.25 * spacing + 2
    ]
    equals = None
    for a in bars_found:
        for b in bars_found:
            if b is a or abs(a[0] - b[0]) > 3 or not 0.15 * spacing <= b[1] - a[1] <= 0.6 * spacing:
                continue
            equals = (a[0] + max(a[2], b[2]), (a[1] + b[1] + b[3]) / 2.0)
            break
        if equals:
            break
    if equals is None:
        return None
    right, middle = equals
    digits = []
    for index in range(1, count):
        x, y, w, h, _area = (int(value) for value in stats[index])
        if x < right or x > right + 5.0 * spacing or abs(y + h / 2 - middle) > 0.6 * spacing:
            continue
        if not 0.5 * spacing <= h <= 2.0 * spacing:
            continue
        digits.append((x, x + w, index))
    if not digits:
        return None
    # 一个数字断成两块（9 的圈和尾巴）：横向重叠的块并成一个字。
    merged: list[list[int]] = []
    for x, end, index in sorted(digits):
        if merged and x < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1].append(index)
        else:
            merged.append([x, end, index])
    digits = []
    for x, end, *indices in merged:
        mask = np.isin(labels, indices)
        rows = np.flatnonzero(mask[:, x:end].any(axis=1))
        digits.append((x, mask[rows[0] : rows[-1] + 1, x:end]))
    # TAB 数字（无衬线）和小节号数字（有衬线）两套字形一起比：有的视频速度用衬线体，8 和 2 只靠无衬线模板分不开。
    from video2score.numbers import _DIGIT_BANKS

    bank = [(label, template) for label, template in tab_bank() if label.isdigit()] + [(str(digit), template) for digit, template in _DIGIT_BANKS]
    text = ""
    for _x, piece in sorted(digits, key=lambda item: item[0]):
        canvas = _fit_glyph(piece)
        # 有圈的先按圈数挑候选：8 两个圈，0、4、6、9 一个（8 和 2 的相关值常常差不多）。圈可能没封口，没圈的不限。
        allowed = _HOLES.get(_holes(piece), "0123456789")
        label, score = max(
            ((label, _glyph_corr(canvas, template)) for label, template in bank if label in allowed), key=lambda item: item[1]
        )
        if score < 0.4:
            return None
        text += label
    value = int(text)
    return value if 20 <= value <= 400 else None


_HOLES = {1: "04689", 2: "8"}


def _holes(piece: np.ndarray) -> int:
    """字形里围住的空白有几块（不和外边相连的背景连通块）。"""
    padded = np.pad(~piece, 1, constant_values=True).astype(np.uint8)
    count, labels = cv2.connectedComponents(padded, connectivity=4)
    outside = labels[0, 0]
    return sum(1 for index in range(1, count) if index != outside and (labels == index).sum() >= 2)
