"""行首的谱号、调号（TAB 谱号的三个字母）：从第一小节切下来，排版时放到每一行开头。

视频里本来就在行首的小节自带这一段，认出来以后排版时切掉，免得一行里出现两次。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from video2score.clean import CleanMeasure
from video2score.tidy import geometry


@dataclass
class Header:
    """行首那一段图（白底黑字，和它所在小节同一线距），以及它在原小节里的宽度。"""

    image: np.ndarray
    spacing: float
    staff_top: float


def extract_header(measures: list[CleanMeasure]) -> Header | None:
    """从最前面几个小节里找第一个带谱号的，切出谱号 + 调号（到拍号之前）。"""
    for measure in measures[:3]:
        end = _header_end(measure)
        if end is not None:
            image = measure.image[:, measure.left_bar : end].copy()
            # 只留每组谱线上下 1.5 个线距以内：上面的小节号、速度记号不要。
            geo = geometry(measure.staff_top, measure.lines, measure.layout, measure.kind, measure.spacing, measure.left_bar, measure.right_bar)
            keep = np.zeros(image.shape[0], bool)
            for ys, _is_tab in geo.groups:
                keep[max(0, int(ys[0] - 1.5 * measure.spacing)) : int(ys[-1] + 1.5 * measure.spacing) + 1] = True
            image[~keep] = 255
            return Header(image=image, spacing=measure.spacing, staff_top=measure.staff_top)
    return None


def content_start(measure: CleanMeasure, header: Header) -> int | None:
    """这一小节左端要是和行首那一段一样（视频里的行首小节），返回切掉它以后内容开始的列。"""
    if abs(measure.spacing - header.spacing) > 0.08 * header.spacing:
        return None
    width = header.image.shape[1]
    top = int(round(header.staff_top))
    height = header.image.shape[0]
    offset = int(round(measure.staff_top)) - top
    if offset < 0 or offset + height > measure.image.shape[0]:
        return None
    template = _symbols(header.image, header.spacing)
    if template.sum() < 20:
        return None
    best = -1.0
    for shift in range(-3, 4):
        x0 = measure.left_bar + shift
        if x0 < 0 or x0 + width > measure.image.shape[1]:
            continue
        patch = _symbols(measure.image[offset : offset + height, x0 : x0 + width], header.spacing)
        overlap = float((patch * template).sum())
        score = 2 * overlap / float(patch.sum() + template.sum() + 1e-6)
        best = max(best, score)
    return measure.left_bar + width if best >= 0.6 else None


def key_fifths(measures: list[CleanMeasure]) -> int | None:
    """调号：升号数为正、降号数为负。从最前面几个带谱号的小节里，数谱号后面的升降号。

    谱号是第一段符号（和 _header_end 一样按列分段）；它后面到调号结束，只留又高又细的竖笔（谱线是横的，不碍事）：
    升号的两根竖笔挨得很近（不到 0.6 个线距），降号只有一根。纯 TAB 谱没有调号，返回 None。
    """
    for measure in measures[:3]:
        end = _header_end(measure)
        if end is None:
            continue
        geo = geometry(measure.staff_top, measure.lines, measure.layout, measure.kind, measure.spacing, measure.left_bar, measure.right_bar)
        ys, is_tab = geo.groups[0]
        if is_tab:
            return None
        spacing = measure.spacing
        y0 = max(0, int(ys[0] - 1.5 * spacing))
        y1 = int(ys[-1] + 1.5 * spacing) + 1
        ink = (measure.image[y0:y1] < 128).astype(np.uint8)
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(5, int(round(1.5 * spacing)))), np.uint8))
        symbols = cv2.subtract(ink, lines)
        start = measure.left_bar + max(2, int(round(0.25 * spacing)))
        segments = _segments(symbols[:, start:end].sum(axis=0) > 0, max(1, int(round(0.3 * spacing))))
        if not segments:
            return 0
        clef_end = start + segments[0][1]
        # 竖笔只有一两像素宽，有的被抗锯齿压淡：数竖笔用浅一些的阈值。
        faint = (measure.image[y0:y1, clef_end:end] < 200).astype(np.uint8)
        strokes = cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((max(5, int(round(1.2 * spacing))), 1), np.uint8))
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(strokes, connectivity=8)
        xs = sorted(float(stats[index, cv2.CC_STAT_LEFT]) for index in range(1, count))
        if not xs:
            return 0
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        paired = sum(1 for gap in gaps if gap < 0.6 * spacing)
        # 升号：竖笔两两成对。
        if paired and paired * 2 >= len(xs) - 1:
            return (len(xs) + 1) // 2
        return -len(xs)
    return None


def erase_label_after_header(measure: CleanMeasure, start: int) -> None:
    """行首小节的号，视频里印在谱号、调号、拍号后面，超出了 tidy 抹号的范围：
    在行首结束处左 1 右 4.5 个线距（号可能在拍号后面）、谱表上方 3 个线距以内，抹掉数字大小的块（速度记号的音符有符干，更高）。"""
    spacing = measure.spacing
    image = measure.image
    y0 = max(0, int(measure.staff_top - 3.0 * spacing))
    y1 = int(measure.staff_top - 0.2 * spacing)
    x0 = max(0, int(start - 1.0 * spacing))
    x1 = min(image.shape[1], int(start + 4.5 * spacing))
    if y1 <= y0 or x1 <= x0:
        return
    ink = (image[:y1] < 128).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for index in range(1, count):
        x, y, w, h, _area = (int(value) for value in stats[index])
        if y < y0 or x < x0 or x + w > x1 or h > 1.1 * spacing or h < 0.4 * spacing or w > 0.9 * spacing:
            continue
        image[:y1][labels == index] = 255


def _symbols(image: np.ndarray, spacing: float) -> np.ndarray:
    """墨迹里去掉谱线（长横线）和小节线（长竖线），只剩谱号、调号这些符号。每小节都有的谱线不能拿来比像不像。"""
    ink = (image < 128).astype(np.uint8)
    length = max(5, int(round(1.5 * spacing)))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    bars = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((int(round(3.5 * spacing)), 1), np.uint8))
    return cv2.subtract(cv2.subtract(ink, lines), bars).astype(np.float32)


def _header_end(measure: CleanMeasure) -> int | None:
    """谱号 + 调号的右端（小节图里的列）。第一组谱线（五线谱或纯 TAB）里按列找符号段：

    第一段贴着左边小节线、够高，是谱号；后面挨得很近的窄段是调号；遇到拍号、或者空出一个线距以上就停。
    """
    geo = geometry(measure.staff_top, measure.lines, measure.layout, measure.kind, measure.spacing, measure.left_bar, measure.right_bar)
    spacing = measure.spacing
    ys, is_tab = geo.groups[0]
    y0 = max(0, int(ys[0] - 1.0 * spacing))
    y1 = min(measure.image.shape[0], int(ys[-1] + 1.0 * spacing) + 1)
    ink = (measure.image[y0:y1] < 128).astype(np.uint8)
    # 抹掉谱线（长横线）和小节线附近，只留符号。
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(5, int(round(1.5 * spacing)))), np.uint8))
    symbols = cv2.subtract(ink, lines)
    start = measure.left_bar + max(2, int(round(0.25 * spacing)))
    columns = symbols[:, start : measure.right_bar].sum(axis=0) > 0
    segments = _segments(columns, max(1, int(round(0.3 * spacing))))
    if not segments:
        return None
    first = segments[0]
    heights = [_height(symbols[:, start + a : start + b]) for a, b in segments]
    # 谱号：离左边小节线不到 1.5 个线距、高过 2 个线距（TAB 的三个字母叠起来也有这么高）。
    if first[0] > 1.5 * spacing or heights[0] < 2.0 * spacing:
        return None
    end = first[1]
    if is_tab:
        # TAB 谱号的三个字母后面常紧挨着拍号，连成一段：在第一段里碰到粗体的列（拍号）就切开。
        size = max(2, int(round(0.2 * spacing)))
        bold_columns = cv2.erode(symbols, np.ones((size, size), np.uint8))[:, start + first[0] : start + first[1]].any(axis=0)
        columns_hit = np.flatnonzero(bold_columns)
        if columns_hit.size:
            # 在字母和拍号之间墨最少的那一列切开（拍号数字左边的细笔画不算粗体，不能只按粗体切）。
            counts = symbols[:, start + first[0] : start + first[1]].sum(axis=0)
            lo = int(round(0.3 * spacing))
            hi = max(lo + 1, int(columns_hit[0]))
            end = first[0] + lo + int(np.argmin(counts[lo:hi]))
        return start + end + max(2, int(round(0.2 * spacing)))
    for (a, b), height in zip(segments[1:], heights[1:]):
        if a - end > 1.2 * spacing or b - a > 8.0 * spacing:
            break
        # 拍号是粗体数字，腐蚀掉细笔画后还剩一大半；升降号笔画细，剩得很少。碰到拍号就停。
        # 有的视频升号的横杠画得很粗，也剩不少：拍号是上下两个数字、占满整个谱表（4 个线距），升降号没这么高。
        if _bold(symbols[:, start + a : start + b], spacing) > 0.3 and height >= 3.3 * spacing:
            break
        end = b
    return start + end + max(2, int(round(0.3 * spacing)))


def _bold(patch: np.ndarray, spacing: float) -> float:
    """腐蚀掉 0.2 个线距以后还剩多少墨（比例）。"""
    total = int(patch.sum())
    if total == 0:
        return 0.0
    size = max(2, int(round(0.2 * spacing)))
    return float(cv2.erode(patch, np.ones((size, size), np.uint8)).sum()) / total


def _segments(columns: np.ndarray, gap: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for index in np.flatnonzero(columns):
        index = int(index)
        if segments and index - segments[-1][1] <= gap:
            segments[-1] = (segments[-1][0], index + 1)
        else:
            segments.append((index, index + 1))
    return segments


def _height(patch: np.ndarray) -> int:
    rows = np.flatnonzero(patch.any(axis=1))
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


# 调号里升降号依次落在谱表上的位置（从第一线往下数几个线距，半格是间）。低音谱号、高音谱号各一套。
_SHARP_STEPS = {True: (1.0, 2.5, 0.5, 2.0, 3.5, 1.5, 3.0), False: (0.0, 1.5, -0.5, 1.0, 2.5, 0.5, 2.0)}
_FLAT_STEPS = {True: (3.0, 1.5, 3.5, 2.0, 4.0, 2.5, 4.5), False: (2.0, 0.5, 2.5, 1.0, 3.0, 1.5, 3.5)}


def key_signature(image: np.ndarray, lines: list[float], spacing: float, x0: int, x1: int, bass: bool = True) -> tuple[int, int] | None:
    """小节开头（x0 是左小节线）印的调号：(升号数为正、降号数为负, 调号右端的列)；没有调号或认不准返回 None。

    行中转调时 GP 在小节线后面直接印新调号（横向滚动的谱没有行首）。行首带谱号的先跳过谱号。
    只认按调号次序、落在规定高度上的一串升号或降号：
    - 升号是两道挨着（不到 0.6 个线距）、上端差不多齐的竖笔，上下两端的中点是它的位置；还原号的两道竖笔一高一低，不算；
    - 降号是一道竖笔，下端往上 0.8 个线距是它的位置（圆肚的中间），右下方要有圆肚；
    - 最后一个后面空出半个线距以上（小节里临时升降号后面紧挨着音符）。
    """
    top = int(max(0, lines[0] - 2.0 * spacing))
    bottom = int(min(image.shape[0], lines[-1] + 2.0 * spacing))
    x1 = min(x1, image.shape[1])
    if x1 - x0 < 2 * spacing:
        return None
    faint = (image[top:bottom, x0:x1] < 200).astype(np.uint8)
    strokes = cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((max(5, int(round(1.2 * spacing))), 1), np.uint8))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    found = []
    start = 0.0
    for index in sorted(range(1, count), key=lambda item: stats[item, cv2.CC_STAT_LEFT]):
        x, y, w, h = (int(value) for value in stats[index, :4])
        if h > 3.4 * spacing or w > 0.35 * spacing:
            # 小节线、谱号：调号从它们后面开始。
            if x - start <= 2.5 * spacing or not found:
                start = max(start, float(x + w))
            continue
        found.append((x, (y + top - lines[0]) / spacing, (y + h + top - lines[0]) / spacing))
    clef = _clef_end(image, lines, spacing, x0, x1)
    if clef is not None:
        start = max(start, float(clef - x0))
        found = [item for item in found if item[0] >= start]
    if not found or found[0][0] - start > 1.5 * spacing:
        return None
    chain = [found[0]]
    for item in found[1:]:
        if item[0] - chain[-1][0] > 1.1 * spacing:
            break
        chain.append(item)
    ink = (image[int(lines[0]) : int(lines[-1]) + 1, x0:x1] < 128).astype(np.uint8)
    ink = cv2.subtract(ink, cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(5, int(round(1.5 * spacing)))), np.uint8)))

    def clear_after(right: int) -> bool:
        window = ink[:, right + 2 : right + 2 + int(round(0.5 * spacing))]
        return window.size > 0 and not window.any()

    sharps = _sharps(chain, spacing, bass)
    if sharps:
        count_sharps, right = sharps
        if clear_after(right):
            return count_sharps, x0 + right
    flats = _flats(chain, ink_full=(image[top:bottom, x0:x1] < 200), top=top, lines=lines, spacing=spacing, bass=bass)
    if flats:
        count_flats, right = flats
        if clear_after(right):
            return -count_flats, x0 + right
    return None


def _sharps(chain: list[tuple[int, float, float]], spacing: float, bass: bool) -> tuple[int, int] | None:
    steps = _SHARP_STEPS[bass]
    count = 0
    right = 0
    for index in range(0, len(chain) - 1, 2):
        (xa, ta, ba), (xb, tb, bb) = chain[index], chain[index + 1]
        if xb - xa > 0.6 * spacing or abs(ta - tb) > 0.4 or count >= len(steps):
            break
        if abs((min(ta, tb) + max(ba, bb)) / 2 - steps[count]) > 0.3:
            break
        count += 1
        right = xb + int(round(0.3 * spacing))
    return (count, right) if count else None


def _flats(chain: list[tuple[int, float, float]], ink_full: np.ndarray, top: int, lines: list[float], spacing: float, bass: bool) -> tuple[int, int] | None:
    steps = _FLAT_STEPS[bass]
    count = 0
    right = 0
    for x, stroke_top, stroke_bottom in chain:
        if count >= len(steps) or abs(stroke_bottom - 0.8 - steps[count]) > 0.3:
            break
        # 圆肚：竖笔下半截右边 0.6 个线距以内有墨。
        y_bottom = int(lines[0] + stroke_bottom * spacing) - top
        bowl = ink_full[max(0, y_bottom - int(0.8 * spacing)) : y_bottom, x + 2 : x + 2 + int(round(0.6 * spacing))]
        if bowl.size == 0 or bowl.any(axis=0).sum() < 0.3 * spacing:
            break
        count += 1
        right = x + int(round(0.7 * spacing))
    return (count, right) if count else None


def _clef_end(image: np.ndarray, lines: list[float], spacing: float, x0: int, x1: int) -> int | None:
    """左小节线后面 1.5 个线距以内的谱号（连同右边的两个点）右端；没有谱号返回 None。

    抹掉谱线后谱号断成好几块，按列分段（和 _header_end 一样）：第一段又宽又高才是谱号。
    """
    top = int(max(0, lines[0] - 1.0 * spacing))
    bottom = int(min(image.shape[0], lines[-1] + 1.0 * spacing))
    ink = (image[top:bottom, x0 : min(x1, x0 + int(6 * spacing))] < 128).astype(np.uint8)
    ink = cv2.subtract(ink, cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(5, int(round(1.5 * spacing)))), np.uint8)))
    ink = cv2.subtract(ink, cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((int(round(3.5 * spacing)), 1), np.uint8)))
    segments = _segments(ink.sum(axis=0) > 0, max(1, int(round(0.3 * spacing))))
    # 小节线抹不干净时会剩一两列细条，不算。
    segments = [(a, b) for a, b in segments if b - a >= 0.3 * spacing]
    if not segments:
        return None
    a, b = segments[0]
    if a > 1.5 * spacing or b - a < 1.5 * spacing or _height(ink[:, a:b]) < 2.5 * spacing:
        return None
    return x0 + b
