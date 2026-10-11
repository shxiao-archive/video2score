"""读节奏：从五线谱（五线谱+TAB 的谱）读出每个音的时值。

做法：
1. 找符干：谱行上下一带里比 1.8 个线距长的细竖线（小节线除外）。
2. 符干两端哪一头有符头（一团墨），哪一头就是符头端；另一头是自由端。
3. 符头实心还是空心：实心是四分及更短，空心是二分（没有符干的空心头是全音符）。
4. 自由端数符杠 / 符尾：沿符干旁边一列往里数有几道横向的粗墨，0 道是四分，1 道八分，2 道十六分。
5. 符头右边的小圆点是附点。
6. 不挨着任何符干、又像休止符的块，按大小认成各种休止符。

时值用 Fraction，以全音符为 1：四分音符是 1/4。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import cv2
import numpy as np

# 很黑的墨（255 里的深度）：数符杠时只看它。
_SOLID = 135
BASE = {0: Fraction(1, 4), 1: Fraction(1, 8), 2: Fraction(1, 16), 3: Fraction(1, 32)}


@dataclass
class Staff5:
    """五线谱的五条线（小节图里的绝对 y）。"""

    lines: tuple[float, ...]
    spacing: float

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def bottom(self) -> float:
        return self.lines[-1]


@dataclass
class Event:
    """一个时值单位：音符（带符头位置）或休止符。"""

    x: float
    duration: Fraction
    rest: bool = False
    dots: int = 0
    beams: int = 0
    head_y: float | None = None
    hollow: bool = False
    tuplet: Fraction | None = None
    # 连音括号的头尾："start"、"stop"。
    tuplet_mark: str | None = None
    # 跳音：符头另一侧（和符干相反的方向）的小圆点。
    staccato: bool = False
    # 重音记号（符头另一侧）："strong-accent"（"^"）或 "accent"（">"），没有是 None。
    accent: str | None = None
    # 符干朝上（True）、朝下（False），不知道（没有符干、纯 TAB）是 None。
    stem_up: bool | None = None
    # 和后面一个音之间有没有符杠连着；不知道是 None（写谱时按拍子分组）。
    beam_next: bool | None = None
    # 被当成空心符头找出来的 X 形闷音符头：符干接在 X 的尖上，符头中心比找到的位置再往外半个线距。
    crossed: bool = False
    # 符干（x, 上端, 下端），只在读五线谱时临时用。
    stem: tuple[int, int, int] | None = field(default=None, repr=False)
    notes: list = field(default_factory=list)

    @property
    def kind(self) -> str:
        return {
            Fraction(1, 1): "whole",
            Fraction(1, 2): "half",
            Fraction(1, 4): "quarter",
            Fraction(1, 8): "eighth",
            Fraction(1, 16): "16th",
            Fraction(1, 32): "32nd",
        }.get(self.base, "quarter")

    @property
    def base(self) -> Fraction:
        value = self.duration
        if self.tuplet:
            value /= self.tuplet
        return value / (Fraction(3, 2) if self.dots == 1 else Fraction(7, 4) if self.dots == 2 else 1)


def staff5(record: dict) -> Staff5 | None:
    top = float(record["staff_top"])
    lines = [top + float(value) for value in record["lines"]]
    start = 0
    for item in record["layout"]:
        count = int(item[0])
        group = lines[start : start + count]
        start += count
        is_staff = item[2] == "staff" if len(item) > 2 else count == 5
        if is_staff and count == 5:
            return Staff5(lines=tuple(group), spacing=(group[-1] - group[0]) / 4.0)
    return None


def read_staff_rhythm(
    image: np.ndarray,
    staff: Staff5,
    bars: tuple[int, int],
    limit: tuple[float, float] | None = None,
) -> list[Event]:
    """读五线谱一小节的节奏。limit 是可用的上下范围（不要伸进 TAB）。"""
    spacing = staff.spacing
    top = int(max(0, staff.top - 4.0 * spacing))
    bottom = int(min(image.shape[0], staff.bottom + 4.0 * spacing))
    if limit is not None:
        top = max(top, int(limit[0]))
        bottom = min(bottom, int(limit[1]))
    ink = _without_shadows((255 - image[top:bottom]).astype(np.int16), spacing)
    # 数符杠只看很黑的墨：符杠是实心的纯黑横条，背后灰色的背景画会把它"加粗"成两三道。
    solid = (255 - image[top:bottom]) >= _SOLID
    lines = [line - top for line in staff.lines]
    no_lines = _without_staff_lines(ink, lines, spacing)
    left_bar, right_bar = bars
    stems = _stems(no_lines, spacing, left_bar, right_bar)
    events: list[Event] = []
    used = np.zeros_like(no_lines)
    # 符头用没抹线的原图找：压在线上的符头会被抹线从中间切开。谱线很细，开运算自己会去掉。
    heads = _noteheads(ink, spacing, left_bar, right_bar)
    heads = _drop_beam_heads(ink, heads, stems, spacing)
    heads = _check_whole_heads(ink, heads, stems, lines, spacing)
    heads = sorted(heads + _hollow_heads_on_stems(ink, stems, heads, lines, spacing))
    for head in heads:
        event = _head_event(no_lines, ink, head, stems, spacing, used, solid, middle=lines[2])
        if event is None:
            continue
        if event is not None:
            events.append(event)
    events = _merge_chords(events, spacing)
    _link_beams(solid, events, spacing)
    events.extend(_whole_notes(no_lines, lines, spacing, used, left_bar, right_bar, events))
    events.extend(_rests(no_lines, lines, spacing, used, left_bar, right_bar, events))
    events.sort(key=lambda event: event.x)
    # 重音记号的细笔画很淡：用浅一些的门槛找。
    faint = (255 - image[top:bottom]) >= 60
    for event in events:
        if not event.rest and event.head_y is not None and event.stem_up is not None:
            event.accent = _accent(faint, event.x, event.head_y, event.stem_up, spacing)
    for event in events:
        if event.head_y is not None:
            event.head_y += top
    _apply_tuplets(no_lines, events, spacing, top, bars, lines)
    return events


def _without_shadows(darkness: np.ndarray, spacing: float) -> np.ndarray:
    """墨（深浅 110 以上），去掉成大片的中灰：谱带压在视频画面上时，画面里的暗影、色块透过来是中灰的一大片
    （比如一只手的影子盖住半个小节，影子边缘被当成符杠、符头）。音符、谱线是纯黑的，压在影子上也留得住。
    大片 = 用一个半线距的圆盘开运算后还在的中灰；只有比一个符头大得多的块才去，符头、符杠都留着。"""
    ink = darkness >= 110
    middle = (ink & (darkness < 170)).astype(np.uint8)
    size = max(5, int(round(1.5 * spacing)) | 1)
    broad = cv2.morphologyEx(middle, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    if not broad.any():
        return ink
    # 大片中灰的边缘一圈（开运算削掉的那一两像素）也是影子。
    broad = cv2.dilate(broad, np.ones((3, 3), np.uint8)).astype(bool)
    return ink & ~(broad & (darkness < 170))


def _without_staff_lines(ink: np.ndarray, lines: list[float], spacing: float) -> np.ndarray:
    """抹掉谱线（只在线那一两行里抹长横线），加线也抹掉。"""
    mask = ink.astype(np.uint8)
    length = max(9, int(round(1.5 * spacing)))
    long_runs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    band = np.zeros_like(mask)
    reach = max(1, int(round(0.08 * spacing)))
    for line in lines:
        row = int(round(line))
        band[max(0, row - reach) : row + reach + 1] = 1
    # 加线：谱线外、间距一个线距的短横线，也只抹这些行里的横线。
    ledger = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(round(1.1 * spacing))), 1))
    )
    for step in range(1, 5):
        for row in (lines[0] - step * spacing, lines[-1] + step * spacing):
            row = int(round(row))
            if 0 <= row < mask.shape[0]:
                band[max(0, row - reach) : row + reach + 1] = 1
    cleaned = cv2.subtract(mask, (long_runs | ledger) & band)
    # 符干穿过谱线的那一点在抹线时被抹掉了，符干会断成几截。原图里的长竖线补回去。
    tall = max(9, int(round(1.2 * spacing)))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall)))
    return (cleaned | vertical).astype(bool)


def _stems(ink: np.ndarray, spacing: float, left_bar: int, right_bar: int) -> list[tuple[int, int, int]]:
    """细长竖线：(x, 上端, 下端)。紧贴小节线的不算。"""
    tall = max(9, int(round(1.8 * spacing)))
    vertical = cv2.morphologyEx(
        ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall))
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(vertical, connectivity=8)
    stems: list[tuple[int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, _area = (int(value) for value in stats[index])
        if width > max(4, 0.35 * spacing):
            continue
        centre = x + width // 2
        if abs(centre - left_bar) < 0.5 * spacing or abs(centre - right_bar) < 0.5 * spacing:
            continue
        stems.append((centre, y, y + height))
    stems.sort()
    # 播放光标：同一列上几截都很长的竖线（各 2.5 个线距以上），合起来占了截图六成以上的高度（中间被符头、谱线挡住），不是符干。
    cursor: set[int] = set()
    for index, (x, _y0, _y1) in enumerate(stems):
        column = [other for other, (ox, _a, _b) in enumerate(stems) if abs(ox - x) <= 2 and stems[other][2] - stems[other][1] >= 2.5 * spacing]
        covered = sum(stems[other][2] - stems[other][1] for other in column)
        if len(column) >= 2 and covered >= 0.6 * ink.shape[0]:
            cursor.update(column)
    stems = [stem for index, stem in enumerate(stems) if index not in cursor]
    # 升号、还原号：两根挨得很近（不到 0.6 个线距）、又不太长的竖线。都不是符干。
    accidental: set[int] = set()
    for i in range(len(stems) - 1):
        (xa, ya0, ya1), (xb, yb0, yb1) = stems[i], stems[i + 1]
        if xb - xa < 0.6 * spacing and max(ya1 - ya0, yb1 - yb0) < 3.2 * spacing:
            overlap = min(ya1, yb1) - max(ya0, yb0)
            if overlap > 0.5 * min(ya1 - ya0, yb1 - yb0):
                accidental.update((i, i + 1))
    return [stem for index, stem in enumerate(stems) if index not in accidental]


def _noteheads(ink: np.ndarray, spacing: float, left_bar: int, right_bar: int) -> list[tuple[float, float, bool]]:
    """符头：去掉竖线后剩下的椭圆块，约 1.1~1.6 个线距宽、0.6~1.1 个线距高。返回 (x, y, 空心)。

    升号、还原号没有这样的实心椭圆；符杠比符头长得多；都排除在外。
    """
    tall = max(9, int(round(1.5 * spacing)))
    vertical = cv2.morphologyEx(
        ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall))
    )
    rest = cv2.subtract(ink.astype(np.uint8), vertical)
    # 实心符头：用半个线距的圆盘开运算，细笔画（符尾、升号的横杠、连线）去掉，实心块留下。
    disk = max(3, int(round(0.45 * spacing)) | 1)
    solid = cv2.morphologyEx(rest, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk, disk)))
    heads: list[tuple[float, float, bool]] = []
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(solid, connectivity=8)
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if not 0.8 * spacing <= width <= 1.7 * spacing or not 0.5 * spacing <= height <= 1.2 * spacing:
            continue
        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        if cx <= left_bar + 0.5 * spacing or cx >= right_bar - 0.3 * spacing:
            continue
        heads.append((cx, cy, False))
    # 空心符头（二分、全音符）：闭合的小椭圆环，填上洞以后符合符头大小，而且原来中间是空的。
    filled = _fill_holes(rest)
    ring = cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk, disk)))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ring, connectivity=8)
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if not 0.8 * spacing <= width <= 1.8 * spacing or not 0.5 * spacing <= height <= 1.2 * spacing:
            continue
        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        if any(abs(cx - hx) < 0.5 * spacing and abs(cy - hy) < 0.5 * spacing for hx, hy, _h in heads):
            continue
        region = labels[y : y + height, x : x + width] == index
        original = rest[y : y + height, x : x + width].astype(bool)
        if float((region & ~original).mean()) < 0.15:
            continue
        if cx <= left_bar + 0.5 * spacing or cx >= right_bar - 0.3 * spacing:
            continue
        heads.append((cx, cy, True))
    return sorted(heads)


def _drop_beam_heads(
    ink: np.ndarray,
    heads: list[tuple[float, float, bool]],
    stems: list[tuple[int, int, int]],
    spacing: float,
) -> list[tuple[float, float, bool]]:
    """符杠粗的那一段会被当成符头。同一根符干两头各有一个"符头"、相距 2.5 个线距以上时，
    横着连着厚墨（符杠，左边或右边 1.0~1.6 个线距处同一高度都有竖着够厚的墨）的那个是假的。"""
    thick = cv2.erode(ink.astype(np.uint8), np.ones((max(2, int(round(0.3 * spacing))), 1), np.uint8)).astype(bool)

    def on_beam(cx: float, cy: float) -> bool:
        y0, y1 = max(0, int(cy - 0.5 * spacing)), int(cy + 0.5 * spacing) + 1
        for sign in (-1, 1):
            columns = [int(round(cx + sign * step * spacing)) for step in np.linspace(1.0, 1.6, 4)]
            columns = [c for c in columns if 0 <= c < ink.shape[1]]
            if columns and sum(thick[y0:y1, c].any() for c in columns) >= len(columns):
                return True
        return False

    drop: set[int] = set()
    for x, y0, y1 in stems:
        mine = [i for i, (hx, hy, _h) in enumerate(heads) if abs(hx - x) <= 0.9 * spacing and y0 - 0.6 * spacing <= hy <= y1 + 0.6 * spacing]
        if len(mine) < 2:
            continue
        ys = [heads[i][1] for i in mine]
        if max(ys) - min(ys) < 2.5 * spacing:
            continue
        for i in mine:
            if on_beam(heads[i][0], heads[i][1]):
                drop.add(i)
        if len(drop & set(mine)) == len(mine):
            drop -= set(mine)
    # 十六分音符和八分音符连在一组时，十六分那个音的第二道符杠只有短短一截（"半截符杠"），贴在符干另一侧、
    # 符干中段，也会被当成符头（低音区密集的十六分音符，一小节能多出三四个）。真符头在符干一头；和弦的几个符头都在符干同一侧。
    for x, y0, y1 in stems:
        mine = [i for i, (hx, hy, _h) in enumerate(heads) if i not in drop and abs(hx - x) <= 0.9 * spacing and y0 - 0.6 * spacing <= hy <= y1 + 0.6 * spacing]
        ends = [i for i in mine if min(abs(heads[i][1] - y0), abs(heads[i][1] - y1)) <= 0.6 * spacing]
        for i in mine:
            if i in ends:
                continue
            hx, hy, hollow = heads[i]
            if any((hx - x) * (heads[j][0] - x) < 0 and abs(hy - heads[j][1]) >= 1.0 * spacing for j in ends):
                drop.add(i)
            elif not ends and not hollow and y1 - y0 <= 5.0 * spacing:
                # 两头都没有符头（真符头是空心的 X 形闷音符头，这时还没找）：符干中段的实心块只能是半截符杠。
                drop.add(i)
    return [head for i, head in enumerate(heads) if i not in drop]


def _line_rows(height: int, lines: list[float], spacing: float) -> np.ndarray:
    """谱线和上下各四条加线所在的那几行（量空心符头时，环里被线穿过的地方不算墨）。"""
    rows = np.zeros(height, bool)
    reach = max(1, int(round(0.1 * spacing)))
    positions = list(lines) + [lines[0] - k * spacing for k in range(1, 5)] + [lines[-1] + k * spacing for k in range(1, 5)]
    for line in positions:
        row = int(round(line))
        if 0 <= row < height:
            rows[max(0, row - reach) : row + reach + 1] = True
    return rows


def _check_whole_heads(
    ink: np.ndarray,
    heads: list[tuple[float, float, bool]],
    stems: list[tuple[int, int, int]],
    lines: list[float],
    spacing: float,
) -> list[tuple[float, float, bool]]:
    """空心符头要像个椭圆环：连线弧和符干、符杠、两道符尾围出来的一圈也会"填上洞"，按环量一遍，不像就去掉。"""
    grown = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    line_rows = _line_rows(ink.shape[0], lines, spacing)
    kept = []
    for hx, hy, hollow in heads:
        # 靠在符干上的也要量：两道符尾和符干也会围出一个"洞"。
        if hollow:
            scores = [
                _ring_score(ink, grown, line_rows, hx + dx * spacing, hy + dy * spacing, spacing)
                for dx in (-0.2, 0.0, 0.2)
                for dy in (-0.2, 0.0, 0.2)
            ]
            if max((score for score in scores if score is not None), default=-1.0) < 0.8:
                continue
        kept.append((hx, hy, hollow))
    return kept


def _hollow_heads_on_stems(
    ink: np.ndarray,
    stems: list[tuple[int, int, int]],
    heads: list[tuple[float, float, bool]],
    lines: list[float],
    spacing: float,
) -> list[tuple[float, float, bool]]:
    """没配上符头的符干，看它两端有没有空心符头（二分音符）。

    找符头时先减掉了竖线，符干贴着的那一侧环跟着断开；谱线从环中间穿过时环也不闭合，按洞找不到。
    这里直接按椭圆环量：符干朝下时符头在符干上端右边，朝上时在下端左边；环上的墨够多、环里是空的（谱线那几行不算）。
    """
    found: list[tuple[float, float, bool]] = []
    grown = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    line_rows = _line_rows(ink.shape[0], lines, spacing)
    for x, y0, y1 in stems:
        if any(abs(hx - x) < 1.0 * spacing and y0 - 0.8 * spacing <= hy <= y1 + 0.8 * spacing for hx, hy, _h in heads + found):
            continue
        best: tuple[float, float, float] | None = None
        # 符干接在符头中部：朝下的符干上端、朝上的符干下端，和符头中心差不多高（往符干里偏一点）。
        for side, end, shifts in ((1, y0, np.arange(-0.3, 0.21, 0.1)), (-1, y1, np.arange(-0.2, 0.31, 0.1))):
            for dx in np.arange(0.4, 0.8, 0.1):
                for dy in shifts:
                    cx, cy = x + side * dx * spacing, end + dy * spacing
                    score = _ring_score(ink, grown, line_rows, cx, cy, spacing)
                    if score is not None and (best is None or score > best[0]):
                        best = (score, cx, cy)
        if best is not None and best[0] >= 0.85:
            found.append((best[1], best[2], True))
    return found


def _ring_score(
    ink: np.ndarray,
    grown: np.ndarray,
    line_rows: np.ndarray,
    cx: float,
    cy: float,
    spacing: float,
    inner_ink: np.ndarray | None = None,
) -> float | None:
    """(cx, cy) 处一个稍斜的空心符头有多像：环上有墨的比例减去环里有墨的比例。inner_ink 给了就用它量环里。"""
    a, b, tilt = 0.58 * spacing, 0.4 * spacing, np.radians(-20.0)
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    xs = cx + a * np.cos(angles) * np.cos(tilt) - b * np.sin(angles) * np.sin(tilt)
    ys = cy + a * np.cos(angles) * np.sin(tilt) + b * np.sin(angles) * np.cos(tilt)
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    if xi.min() < 0 or yi.min() < 0 or xi.max() >= ink.shape[1] or yi.max() >= ink.shape[0]:
        return None
    ring = float(grown[yi, xi].mean())
    # 中心那一小块也算进环里：闷音的 X 形符头环上也有墨，但交叉点正好在中心。
    radius = max(1, int(round(0.12 * spacing)))
    inside = ink if inner_ink is None else inner_ink
    centre = inside[int(round(cy)) - radius : int(round(cy)) + radius + 1, int(round(cx)) - radius : int(round(cx)) + radius + 1]
    inner = centre[~line_rows[int(round(cy)) - radius : int(round(cy)) + radius + 1]].ravel().tolist() if centre.size else []
    for scale in (0.2, 0.4):
        px, py = np.round(cx + scale * (xs - cx)).astype(int), np.round(cy + scale * (ys - cy)).astype(int)
        keep = ~line_rows[py]
        inner.extend(inside[py[keep], px[keep]].tolist())
    fill = float(np.mean(inner)) if inner else 0.0
    return ring - fill


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """把闭合笔画围起来的洞填上（从边框往里泛洪，没泛到的背景就是洞）。"""
    padded = np.pad(mask.astype(np.uint8), 1)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return (padded | holes)[1:-1, 1:-1]


def _head_event(
    ink: np.ndarray,
    original: np.ndarray,
    head: tuple[float, float, bool],
    stems: list[tuple[int, int, int]],
    spacing: float,
    used: np.ndarray,
    solid: np.ndarray | None = None,
    middle: float | None = None,
) -> Event | None:
    """一个符头：找紧贴它的符干，数符杠，看附点。middle 是五线谱中线的 y。"""
    head_x, head_y, hollow = head
    crossed = False
    stem = None
    for x, y0, y1 in stems:
        if abs(x - head_x) > 0.9 * spacing:
            continue
        if y0 - 0.6 * spacing <= head_y <= y1 + 0.6 * spacing:
            if stem is None or abs(x - head_x) < abs(stem[0] - head_x):
                stem = (x, y0, y1)
    # 实心符头一定有符干。没有符干的实心块是升号的交叉处、连线的粗段，不是音符。
    if stem is None and not hollow:
        return None
    # 附点用原图找：抹线后符尾的尖会断成一个小点，原图里它还连着符干，碰到框边就被排除。
    dots = _dots(original, head_x, head_y, spacing)
    half_w = int(round(0.8 * spacing))
    used[
        max(0, int(head_y - 0.7 * spacing)) : int(head_y + 0.7 * spacing),
        max(0, int(head_x - half_w)) : int(head_x + half_w),
    ] = True
    if stem is None:
        base = Fraction(1, 1) if hollow else Fraction(1, 4)
        beams = 0
    else:
        x, y0, y1 = stem
        used[y0:y1, max(0, x - 2) : x + 3] = True
        # 符头在符干下端附近是符干朝上，自由端在上面。
        stem_up = abs(head_y - y1) < abs(head_y - y0)
        if middle is not None and y1 - y0 > 5.0 * spacing and min(abs(head_y - y0), abs(head_y - y1)) > 1.5 * spacing:
            # 竖线比符干长得多、符头在它中段：符干和播放光标（贯穿上下的竖线）连在一起了。按符头高度定：中线及以上朝下。
            stem_up = head_y > middle + 0.1 * spacing
        free_end = y0 if stem_up else y1
        beams = _count_beams(solid if solid is not None else original, x, free_end, stem_up, spacing, flags_ink=original)
        # 空心符头不会带符杠、符尾：带了的是被当成空心的 X 形闷音符头，按符杠数算。
        if hollow and beams >= 1:
            hollow = False
            crossed = True
        base = Fraction(1, 2) if hollow else BASE.get(min(beams, 3), Fraction(1, 4))
    duration = base * (Fraction(3, 2) if dots == 1 else Fraction(7, 4) if dots >= 2 else 1)
    staccato = stem is not None and _staccato(ink, head_x, head_y, stem_up, spacing)
    return Event(
        x=head_x,
        duration=duration,
        dots=dots,
        beams=beams,
        head_y=head_y,
        hollow=hollow,
        staccato=staccato,
        stem_up=stem_up if stem is not None else None,
        stem=stem,
        crossed=crossed,
    )


def _accent(ink: np.ndarray, head_x: float, head_y: float, stem_up: bool, spacing: float) -> str | None:
    """符头另一侧 1.2~4.5 个线距处的重音记号，约一个线距见方的孤立小块：
    - "strong-accent"：尖朝外（上方是 "^"，下方是 "v"）——靠外的三分之一墨在中间，开口那一头靠外的一行分成左右两段；
    - "accent"：横放的 ">"——尖朝右，最右边的几列墨在中间那一段，开口在左，最左边有一列分成上下两段。"""
    x0, x1 = int(head_x - 1.0 * spacing), int(head_x + 1.0 * spacing)
    if stem_up:
        y0, y1 = int(head_y + 1.2 * spacing), int(head_y + 4.5 * spacing)
    else:
        y0, y1 = int(head_y - 4.5 * spacing), int(head_y - 1.2 * spacing)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(ink.shape[1], x1), min(ink.shape[0], y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    # 细的那一笔常断成几个点：先膨胀一下连起来。
    patch = cv2.dilate(ink[y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
    for index in range(1, count):
        px, py, pw, ph, _area = (int(value) for value in stats[index])
        if px == 0 or py == 0 or px + pw >= patch.shape[1] or py + ph >= patch.shape[0]:
            continue
        mask = labels[py : py + ph, px : px + pw] == index
        # ">" 比 "^" 宽：一个半线距上下。
        if 0.8 * spacing <= pw <= 1.9 * spacing and 0.5 * spacing <= ph <= 1.3 * spacing and _wedge(mask):
            return "accent"
        if not (0.6 * spacing <= pw <= 1.4 * spacing and 0.6 * spacing <= ph <= 1.4 * spacing):
            continue
        third = max(1, ph // 3)
        tip, base = (mask[:third], mask[-third:]) if not stem_up else (mask[-third:], mask[:third])
        columns = np.arange(pw)
        tip_x = columns[tip.any(axis=0)]
        if tip_x.size == 0 or abs(float(tip_x.mean()) - pw / 2.0) > 0.2 * pw:
            continue
        # 开口那一头靠外的几行（离外端 15% 以内，粗的那一笔可能多出一两行）：有一行分成左右两段、中间空开。
        start = int(round(0.85 * (ph - 1)))
        rows = mask[start:] if not stem_up else mask[: ph - start]
        quarter = max(1, pw // 4)
        if any(len(_runs(row)) >= 2 and row[:quarter].any() and row[-quarter:].any() for row in rows):
            return "strong-accent"
    return None


def _wedge(mask: np.ndarray) -> bool:
    """横放的 ">"：比高宽；右边五分之一的墨都在中间那一半高度里，左边五分之一里有一列分成上下两段、各贴着上下边。"""
    height, width = mask.shape
    if width < 1.1 * height:
        return False
    fifth = max(1, width // 5)
    tip_rows = np.flatnonzero(mask[:, -fifth:].any(axis=1))
    if tip_rows.size == 0 or tip_rows[0] < 0.25 * height or tip_rows[-1] > 0.75 * height:
        return False
    quarter = max(1, height // 4)
    return any(
        len(_runs(column)) >= 2 and column[:quarter].any() and column[-quarter:].any() for column in mask[:, :fifth].T
    )


def _staccato(ink: np.ndarray, head_x: float, head_y: float, stem_up: bool, spacing: float) -> bool:
    """符头另一侧 0.5~2 个线距处、左右对齐的一个孤立小圆点（符干朝上点在下面，朝下点在上面）。"""
    x0, x1 = int(head_x - 0.6 * spacing), int(head_x + 0.6 * spacing)
    if stem_up:
        y0, y1 = int(head_y + 0.5 * spacing), int(head_y + 2.0 * spacing)
    else:
        y0, y1 = int(head_y - 2.0 * spacing), int(head_y - 0.5 * spacing)
    # 往外多看半个线距：小点连着的连线尖头、背景图碎块在大一圈的框里就不再是孤立的小点。
    margin = int(round(0.5 * spacing))
    if x0 - margin < 0 or y0 - margin < 0 or x1 + margin > ink.shape[1] or y1 + margin > ink.shape[0]:
        return False
    patch = ink[y0 - margin : y1 + margin, x0 - margin : x1 + margin].astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
    for index in range(1, count):
        px, py, pw, ph, area = (int(value) for value in stats[index])
        # 中心要在原来的框里；碰到大框边的是符干、连线、谱线这些更长的东西。
        cx, cy = px + pw / 2.0 - margin, py + ph / 2.0 - margin
        if not (0 <= cx < x1 - x0 and 0 <= cy < y1 - y0):
            continue
        if px == 0 or py == 0 or px + pw >= patch.shape[1] or py + ph >= patch.shape[0]:
            continue
        # 跳音的点有 0.3~0.4 个线距大；两三个像素的碎渣（连线尾巴被谱线切下的一截）不算。
        if 0.25 * spacing <= max(pw, ph) <= 0.5 * spacing and 0.6 <= pw / max(1, ph) <= 1.6 and area >= 0.5 * pw * ph:
            return True
    return False


def _link_beams(solid: np.ndarray, events: list[Event], spacing: float) -> None:
    """相邻两个带符杠的音（符干同向）之间，两根符干的自由端连成的那条线上一路都是很黑的墨，就是一根符杠连着。"""
    notes = sorted((event for event in events if not event.rest), key=lambda event: event.x)
    for event in notes:
        event.beam_next = False if event.stem is not None else None
    # 只认竖着够厚的墨（符杠约半个线距厚）：谱线很细，符杠贴着谱线走时不能把谱线当成符杠。
    thick = max(2, int(round(0.3 * spacing)))
    heavy = cv2.erode(solid.astype(np.uint8), np.ones((thick, 1), np.uint8)).astype(bool)
    # 一组符杠斜着起头的那一段只有三四行厚，在符干那头数不出符杠：这种音按"三行以上的墨"再看一遍连没连着。
    light = cv2.erode(solid.astype(np.uint8), np.ones((3, 1), np.uint8)).astype(bool)
    for first, second in zip(notes, notes[1:]):
        if first.stem is None or second.stem is None or first.stem_up != second.stem_up:
            continue
        if first.beams < 1 and second.beams < 1:
            continue
        if (first.beams < 1 or second.beams < 1) and (first.hollow or second.hollow):
            continue
        solid = heavy if first.beams >= 1 and second.beams >= 1 else light
        # 第一道符杠贴着符干自由端，往里四分之一个线距是它的中线。
        inward = 0.25 * spacing
        (xa, ya0, ya1), (xb, yb0, yb1) = first.stem, second.stem
        ya = ya0 + inward if first.stem_up else ya1 - inward
        yb = yb0 + inward if second.stem_up else yb1 - inward
        reach = max(1, int(round(0.3 * spacing)))
        hits = total = 0
        for t in np.linspace(0.15, 0.85, 12):
            x = int(round(xa + t * (xb - xa)))
            y = int(round(ya + t * (yb - ya)))
            if not 0 <= x < solid.shape[1]:
                continue
            total += 1
            hits += bool(solid[max(0, y - reach) : y + reach + 1, x].any())
        first.beam_next = total > 0 and hits >= 0.8 * total
        if first.beam_next:
            # 符杠连着的音至少是八分：数不出符杠的那个按一道算。
            for event in (first, second):
                if event.beams < 1:
                    event.beams = 1
                    event.duration = Fraction(1, 8) * (Fraction(3, 2) if event.dots == 1 else Fraction(7, 4) if event.dots >= 2 else 1)


def _merge_chords(events: list[Event], spacing: float) -> list[Event]:
    """同一根符干上的几个符头是一个和弦：横坐标差不到半个线距的合成一个事件。"""
    merged: list[Event] = []
    for event in sorted(events, key=lambda item: item.x):
        if merged and abs(event.x - merged[-1].x) < 0.5 * spacing:
            continue
        merged.append(event)
    return merged


def _blob(ink: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> tuple[float, float, float] | None:
    """一块区域里的墨：(墨的比例, 中心 x, 中心 y)。"""
    x0, x1 = max(0, x0), min(ink.shape[1], x1)
    y0, y1 = max(0, y0), min(ink.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = ink[y0:y1, x0:x1]
    if not patch.any():
        return None
    ys, xs = np.nonzero(patch)
    return float(patch.mean()), float(xs.mean() + x0), float(ys.mean() + y0)


def _note_event(
    ink: np.ndarray,
    original: np.ndarray,
    x: int,
    y0: int,
    y1: int,
    lines: list[float],
    spacing: float,
    used: np.ndarray,
) -> Event | None:
    """一根符干：找符头、数符杠、看附点。"""
    head_w = int(round(1.2 * spacing))
    head_h = int(round(0.9 * spacing))
    # 符头在符干的哪一端：符干朝上时符头在左下，朝下时在右上。
    candidates = [
        ("down_end", _blob(ink, x - head_w, x + 1, y1 - head_h, y1 + head_h // 2)),
        ("up_end", _blob(ink, x, x + head_w + 1, y0 - head_h // 2, y0 + head_h)),
    ]
    best = max(candidates, key=lambda item: -1 if item[1] is None else item[1][0])
    if best[1] is None or best[1][0] < 0.2:
        return None
    fill, head_x, head_y = best[1]
    stem_up = best[0] == "down_end"
    free_end = y0 if stem_up else y1
    # 空心符头：符头框中心附近是白的。
    hollow = _hollow(ink, head_x, head_y, spacing)
    beams = _count_beams(original, x, free_end, stem_up, spacing)
    dots = _dots(ink, head_x, head_y, spacing)
    used[max(0, y0 - head_h) : y1 + head_h, max(0, x - head_w) : x + head_w + 1] = True
    if hollow:
        base = Fraction(1, 2)
    else:
        base = BASE.get(min(beams, 3), Fraction(1, 4))
    duration = base * (Fraction(3, 2) if dots == 1 else Fraction(7, 4) if dots >= 2 else 1)
    return Event(x=head_x, duration=duration, dots=dots, beams=beams, head_y=head_y, hollow=hollow)


def _hollow(ink: np.ndarray, head_x: float, head_y: float, spacing: float) -> bool:
    radius = max(1, int(round(0.18 * spacing)))
    cx, cy = int(round(head_x)), int(round(head_y))
    core = ink[max(0, cy - radius) : cy + radius + 1, max(0, cx - radius) : cx + radius + 1]
    return core.size > 0 and float(core.mean()) < 0.35


def _count_beams(
    ink: np.ndarray,
    x: int,
    free_end: int,
    stem_up: bool,
    spacing: float,
    flags_ink: np.ndarray | None = None,
) -> int:
    """在符干自由端附近，符干左右两侧各看一列，数有几道符杠或符尾。

    用没抹谱线的原图：符杠常压在谱线上，抹线会把它切成两条细线。
    按每段墨的长度折算：一道符杠约 0.45 个线距厚，两道之间空 0.25 个线距；
    谱线穿过两道符杠的空当时，两道会连成一段，也按长度拆开。只有一两像素厚的是谱线，不算。
    """
    span = int(round(1.8 * spacing))
    if stem_up:
        rows = slice(max(0, free_end - 2), min(ink.shape[0], free_end + span))
    else:
        rows = slice(max(0, free_end - span), min(ink.shape[0], free_end + 3))
    offset = max(3, int(round(0.35 * spacing)))
    unit = 0.7 * spacing
    best = 0
    # 一组符杠的头一个、最后一个音，符杠斜着起头，紧挨符干那一列只有薄薄一层：数不到再往外看一列。
    for columns in ((x - offset, x + offset), (x - 2 * offset, x + 2 * offset)):
        for column in columns:
            if not 0 <= column < ink.shape[1]:
                continue
            profile = ink[rows, max(0, column - 1) : column + 2].all(axis=1)
            count = 0
            for length in _runs(profile):
                if length < max(3, 0.3 * spacing):
                    continue
                count += max(1, int(round((length + 0.25 * spacing) / unit)))
            best = max(best, count)
        if best:
            break
    if best > 1:
        return best
    # 两道细符尾挨在一起时粗墨段只数得出一段：按连着符干的细笔段再数一遍，取多的。
    flags = _flag_count(flags_ink if flags_ink is not None else ink, x, rows, offset, spacing)
    return max(best, flags) if best else flags


def _flag_count(ink: np.ndarray, x: int, rows: slice, offset: int, spacing: float) -> int:
    # 单独一个音的符尾：细细的一道弯，挂在符干右边，三列同时有墨的行很少。只看右边一列，数有几段。
    # 符尾是从符干上长出来的；不连着符干的（旁边的滑音线、下一个音的升号）不算。
    # 符尾细，抗锯齿后偏灰：用普通门槛的墨（flags_ink）数。
    column = x + offset
    if not 0 <= column < ink.shape[1]:
        return 0
    window = ink[rows, max(0, x - 1) : column + 2].astype(np.uint8)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
    stem = set(np.unique(labels[:, : min(3, labels.shape[1])])) - {0}
    attached = np.isin(labels[:, column - max(0, x - 1)], list(stem))
    runs = [length for length in _runs(attached) if length >= max(3, 0.2 * spacing)]
    return min(len(runs), 3)


def _runs(profile: np.ndarray) -> list[int]:
    padded = np.concatenate(([0], profile.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts, ends = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
    return list(ends - starts)


def _dots(ink: np.ndarray, head_x: float, head_y: float, spacing: float) -> int:
    """符头右边、同一高度附近的小圆点：又小又圆又实，不碰别的笔画。"""
    # 符头压在线上时，附点写在上面的间里（高出约 0.6 个线距），所以上下各看 0.95 个线距；
    # 双附点的第二个点在 2 个线距开外。
    x0 = int(round(head_x + 0.6 * spacing))
    x1 = int(round(head_x + 2.5 * spacing))
    y0 = int(round(head_y - 0.95 * spacing))
    y1 = int(round(head_y + 0.95 * spacing))
    patch = ink[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)].astype(np.uint8)
    if patch.size == 0:
        return 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
    dots = 0
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        # 碰到框边的是伸进来的别的笔画（下一个音的升号、符杠），不是附点。
        if x == 0 or y == 0 or x + width >= patch.shape[1] or y + height >= patch.shape[0]:
            continue
        round_enough = 0.6 <= width / max(1, height) <= 1.6
        if 0.2 * spacing <= max(width, height) <= 0.55 * spacing and round_enough and area >= 0.55 * width * height:
            dots += 1
    if not dots:
        dots = _touching_dot(patch, head_x - max(0, x0), spacing)
    return min(dots, 2)


def _touching_dot(patch: np.ndarray, head_x: float, spacing: float) -> int:
    """附点贴在下一个音的还原号、升号竖笔上（GP 排得挤时）：连成一块、碰到框边就数不到。
    用比细笔画粗的圆核开运算，把一两像素粗的竖笔、谱线去掉，剩下贴着符头（中心 1.5 个线距内）、附点大小的实心圆块；
    它左边是空的、右边紧挨着一道长竖笔才算。符尾和滑音斜线交叉处也会剩下一个圆块，但它左边连着斜线、右边没有竖笔。"""
    size = max(3, 2 * int(round(0.15 * spacing)) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    ink = patch > 0
    reach = max(2, int(round(0.4 * spacing)))
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if x == 0 or y == 0 or x + width >= patch.shape[1] or y + height >= patch.shape[0]:
            continue
        if centroids[index][0] - head_x > 1.5 * spacing:
            continue
        if not (0.25 * spacing <= max(width, height) <= 0.55 * spacing and 0.7 <= width / max(1, height) <= 1.4 and area >= 0.6 * width * height):
            continue
        # 开运算会把圆块削小一圈：隔开一列再看左边。
        if ink[y : y + height, max(0, x - 4) : max(0, x - 1)].sum() > 1:
            continue
        # 竖笔可能就压在圆块右半边上（开运算后圆块连着一截竖笔），从圆块中间往右看。
        columns = ink[:, x + width // 2 : x + width + reach]
        if columns.size and max(_longest_true_run(column) for column in columns.T) >= 1.0 * spacing:
            return 1
    return 0


def _longest_true_run(values: np.ndarray) -> int:
    runs = _runs(values)
    return max(runs) if runs else 0


def _whole_notes(ink, lines, spacing, used, left_bar, right_bar, events) -> list[Event]:
    """没有符干的空心椭圆：全音符。"""
    return []


def _rests(
    ink: np.ndarray,
    lines: list[float],
    spacing: float,
    used: np.ndarray,
    left_bar: int,
    right_bar: int,
    events: list[Event],
) -> list[Event]:
    """不挨着符干的块，按形状认休止符。

    - 全休止 / 二分休止：线上的实心小矩形（宽约 1 个线距、高约半个）。挂在第四线下是全休止，坐在第三线上是二分休止。
    - 四分休止：竖着的锯齿，高约 2.5~3 个线距。
    - 八分、十六分休止：一根斜杆加一、两个圆头，高约 1.5~2.5 个线距。
    """
    free = ink & ~used
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    # 没抹干净的一两像素高的谱线残渣不要：拼休止符的上下两截时会把它也并进来，把休止符撑宽。
    thin = max(2, int(round(0.12 * spacing)))
    flat = [index for index in range(1, count) if stats[index][cv2.CC_STAT_HEIGHT] <= thin and stats[index][cv2.CC_STAT_WIDTH] >= 3 * stats[index][cv2.CC_STAT_HEIGHT]]
    if flat:
        free = free & ~np.isin(labels, flat)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    count, labels, stats = _join_cut_pieces(count, labels, stats, max(3, int(round(0.2 * spacing))))
    top, bottom = lines[0], lines[-1]
    rests: list[Event] = []
    line_rows = {int(round(line)) + offset for line in lines for offset in (-1, 0, 1)}
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        # 谱线没抹干净的一截粘在休止符旁边，把它撑宽：宽度只按不在谱线上的那些行量。
        mask = labels[y : y + height, x : x + width] == index
        keep = [row for row in range(height) if y + row not in line_rows]
        columns = np.flatnonzero(mask[keep].any(axis=0)) if keep else np.array([], int)
        if columns.size and columns[-1] - columns[0] + 1 < width:
            x, width = x + int(columns[0]), int(columns[-1] - columns[0] + 1)
        centre_x = x + width / 2.0
        if centre_x <= left_bar + 0.8 * spacing or centre_x >= right_bar - 0.3 * spacing:
            continue
        # 紧挨着音符的块是它的附点、连线碎块；音符左边 1.8 个线距内是写升降号的地方。都不是休止符。
        # 音符右边 1.6 个线距内常有抹线后和符干断开的符尾，也不是休止符。
        if any(-1.8 * spacing < centre_x - event.x < 1.6 * spacing for event in events):
            continue
        if y + height < top - 0.5 * spacing or y > bottom + 0.5 * spacing:
            continue
        fill = area / max(1, width * height)
        if 0.8 * spacing <= width <= 1.6 * spacing and 0.3 * spacing <= height <= 0.7 * spacing and fill >= 0.7:
            centre_y = y + height / 2.0
            # 挂在第四线（从上数第二条）下面是全休止，坐在第三线上是二分休止。
            whole = centre_y < lines[1] + 0.5 * spacing
            rests.append(_rest(free, x, y, width, height, spacing, centre_x, Fraction(1, 1) if whole else Fraction(1, 2)))
            continue
        # 十六分休止也有两个半线距高，但比四分休止宽（1.2 个线距以上）、矮一点、墨稀：按圆头数。
        if 2.2 * spacing <= height < 2.85 * spacing and 1.2 * spacing <= width <= 1.5 * spacing and fill < 0.36:
            knobs = _knobs(labels[y : y + height, x : x + width] == index, spacing)
            duration = Fraction(1, 16) if knobs >= 2 else Fraction(1, 8)
            rests.append(_rest(free, x, y, width, height, spacing, centre_x, duration))
            continue
        if 2.2 * spacing <= height <= 3.4 * spacing and 0.5 * spacing <= width <= 1.3 * spacing:
            rests.append(_rest(free, x, y, width, height, spacing, centre_x, Fraction(1, 4)))
            continue
        if 1.3 * spacing <= height <= 2.6 * spacing and 0.5 * spacing <= width <= 1.4 * spacing and fill < 0.55:
            knobs = _knobs(labels[y : y + height, x : x + width] == index, spacing)
            if knobs >= 1:
                rests.append(_rest(free, x, y, width, height, spacing, centre_x, BASE[min(knobs, 3)]))
    return rests


def _rest(
    ink: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    spacing: float,
    centre_x: float,
    duration: Fraction,
) -> Event:
    """休止符，右边一个线距内、上半截高度上有个小圆点就是附点休止。"""
    x0, x1 = x + width, x + width + int(round(1.0 * spacing))
    y0, y1 = max(0, y - int(round(0.4 * spacing))), y + max(int(round(0.8 * spacing)), height // 2)
    patch = ink[y0:y1, x0:x1].astype(np.uint8)
    dots = 0
    if patch.size:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
        for index in range(1, count):
            px, py, pw, ph, area = (int(value) for value in stats[index])
            if px + pw >= patch.shape[1] or py == 0 or py + ph >= patch.shape[0]:
                continue
            if 0.15 * spacing <= max(pw, ph) <= 0.55 * spacing and 0.6 <= pw / max(1, ph) <= 1.6 and area >= 0.5 * pw * ph:
                dots = 1
                break
    return Event(x=centre_x, duration=duration * (Fraction(3, 2) if dots else 1), rest=True, dots=dots)


def _join_cut_pieces(
    count: int,
    labels: np.ndarray,
    stats: np.ndarray,
    gap: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """抹谱线把休止符切成上下几截：左右重叠、上下只隔着一条谱线的块并成一个。"""
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _a in stats]
    # 一两个像素的碎点不参与拼接：它会同时"挨着"好几块，把不相干的块串到一起。
    tiny = {index for index in range(1, count) if int(stats[index][4]) < 4}
    for i in range(1, count):
        for j in range(i + 1, count):
            if i in tiny or j in tiny:
                continue
            ax0, ay0, ax1, ay1 = boxes[i]
            bx0, by0, bx1, by1 = boxes[j]
            overlap = min(ax1, bx1) - max(ax0, bx0)
            if overlap < 0.3 * min(ax1 - ax0, bx1 - bx0):
                continue
            # 上下分开、中间只隔着一条谱线的才拼；上下叠在一起的大块（符杠、符干）不拼。
            if -1 <= max(ay0, by0) - min(ay1, by1) <= gap:
                parent[find(j)] = find(i)
    roots = sorted({find(index) for index in range(1, count)})
    if len(roots) == count - 1:
        return count, labels, stats
    renumber = np.zeros(count, np.int32)
    for new, root in enumerate(roots, start=1):
        renumber[root] = new
    for index in range(1, count):
        renumber[index] = renumber[find(index)]
    merged = renumber[labels]
    joined = np.zeros((len(roots) + 1, 5), np.int32)
    for new in range(1, len(roots) + 1):
        members = [index for index in range(1, count) if renumber[index] == new]
        x0 = min(boxes[index][0] for index in members)
        y0 = min(boxes[index][1] for index in members)
        x1 = max(boxes[index][2] for index in members)
        y1 = max(boxes[index][3] for index in members)
        joined[new] = (x0, y0, x1 - x0, y1 - y0, sum(int(stats[index][4]) for index in members))
    return len(roots) + 1, merged, joined


def _knobs(mask: np.ndarray, spacing: float) -> int:
    """八分、十六分休止左边的圆头个数：左半边从上往下，粗的墨段有几段。"""
    left = mask[:, : max(1, mask.shape[1] // 2)]
    profile = left.sum(axis=1) >= max(2, 0.25 * spacing)
    count = len([length for length in _runs(profile) if length >= max(2, 0.2 * spacing)])
    if count < 2:
        # 斜杆很斜时圆头偏到右半边去了：改看整行，比斜杆（0.2 个线距粗）宽一倍的几段是圆头。
        wide = mask.sum(axis=1) >= max(4, 0.4 * spacing)
        count = max(count, len([length for length in _runs(wide) if length >= max(2, 0.2 * spacing)]))
    return count


def _apply_tuplets(
    ink: np.ndarray,
    events: list[Event],
    spacing: float,
    top: int,
    bars: tuple[int, int],
    lines: list[float],
) -> None:
    """连音：符杠（或符头）外侧单独的一个斜体数字 3（5、6），常带一道两头下弯的括号。

    有括号时括号底下的音是一组；没有括号时取横坐标最靠近数字的 n 个音。
    数字在谱表外、离符头 1.5~5.5 个线距（在符干外头）；两边小节线旁边的小节号不算。
    """
    notes = [event for event in events if not event.rest and event.head_y is not None]
    if len(notes) < 2:
        return
    from video2score.numbers import _fit_glyph, _glyph_corr

    mask = ink.astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    thin = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(round(0.8 * spacing))), 1))
    )
    for index in range(1, count):
        x, y, width, height, _area = (int(value) for value in stats[index])
        if not 0.6 * spacing <= height <= 1.6 * spacing or not 0.35 * spacing <= width <= 1.2 * spacing:
            continue
        cx, cy = x + width / 2.0, y + height / 2.0 + top
        # 连音数字写在谱表外；谱表里面的粗体数字是拍号。
        if lines[0] - 0.3 * spacing < y + height / 2.0 < lines[-1] + 0.3 * spacing:
            continue
        # 小节号印在谱表上方、小节线旁边（左边是本小节的号，可能往右排到 5.5 个线距；右边露出下一小节的号）。
        above = y + height / 2.0 < lines[0]
        if above and (cx < bars[0] + 5.5 * spacing or cx > bars[1] - 2.5 * spacing):
            continue
        # 左右 3 个线距内有一个音的符头离它 1.5~5.5 个线距就行；但正底下（1 个线距内）有符头贴着它的不算
        # （那是别的记号）。旁边一组符干朝下的音，符头可能就在数字斜下方，不能拿最近的那个一票否决。
        around = [(abs(note.x - cx), abs(cy - (note.head_y or cy))) for note in notes if abs(note.x - cx) < 3 * spacing]
        # 很低（或很高）的音符干长，数字在符干头外面、离符头远：这时按符干头量，0.5~4 个线距（符干头在谱表里时，GP 把括号抬到谱表外）。
        ends = [_stem_end(note, top) for note in notes if abs(note.x - cx) < 3 * spacing and note.stem is not None]
        beyond = any((cy < end if up else cy > end) and 0.5 * spacing <= abs(cy - end) <= 4.0 * spacing for end, up in ends)
        if not beyond and not any(1.5 * spacing <= distance <= 5.5 * spacing for _dx, distance in around):
            continue
        if any(dx < 1.0 * spacing and distance < 1.5 * spacing for dx, distance in around):
            continue
        canvas = _fit_glyph(labels[y : y + height, x : x + width] == index)
        scores: dict[str, float] = {}
        for label, template in _tuplet_bank():
            scores[label] = max(scores.get(label, -1.0), _glyph_corr(canvas, template))
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        label, score = ranked[0]
        span = _bracket(thin, x, x + width, y, height, spacing)
        clear = label in "356" and score >= 0.6 and score - ranked[1][1] >= 0.05
        if not clear and span is None:
            continue
        # 两边有括号时，数字认不清也当 3（斜体的 3 和模板差得远）。
        number = int(label) if clear else 3
        if span is not None:
            group = [event for event in events if span[0] - 0.4 * spacing <= event.x <= span[1] + 0.4 * spacing]
        else:
            group = _nearest_run(events, cx, number)
        group = [event for event in group if event.tuplet is None]
        if len(group) < 2:
            continue
        normal = 1
        while normal * 2 < number:
            normal *= 2
        ratio = Fraction(normal, number)
        for event in group:
            event.tuplet = ratio
            event.duration *= ratio
        group[0].tuplet_mark = "start"
        group[-1].tuplet_mark = "stop"


def _stem_end(note: Event, top: int) -> tuple[float, bool]:
    """符干不连符头的那一头的 y（整图坐标），以及符干是不是朝上。"""
    _x, y0, y1 = note.stem
    return (y0 if note.stem_up else y1) + top, bool(note.stem_up)


def _bracket(thin: np.ndarray, x0: int, x1: int, y: int, height: int, spacing: float) -> tuple[int, int] | None:
    """数字左右同一高度上的细横线：连音括号。返回括号的左右端。"""
    rows = slice(max(0, y - int(round(0.3 * spacing))), min(thin.shape[0], y + height + int(round(0.3 * spacing))))
    band = thin[rows].any(axis=0)
    reach = int(round(1.0 * spacing))
    left_side = band[max(0, x0 - reach) : x0]
    right_side = band[x1 : x1 + reach]
    if not left_side.any() or not right_side.any():
        return None
    left = x0 - reach + int(np.flatnonzero(left_side)[0]) if x0 - reach >= 0 else int(np.flatnonzero(left_side)[0])
    while left > 0 and band[left - 1]:
        left -= 1
    right = x1 + int(np.flatnonzero(right_side)[-1])
    while right + 1 < len(band) and band[right + 1]:
        right += 1
    return left, right


def _nearest_run(events: list[Event], x: float, number: int) -> list[Event]:
    """横坐标以 x 为中心的连续 number 个事件。"""
    if len(events) < number:
        return []
    best = min(range(len(events) - number + 1), key=lambda start: abs(np.mean([e.x for e in events[start : start + number]]) - x))
    return events[best : best + number]


@lru_cache(maxsize=1)
def _tuplet_bank() -> tuple[tuple[str, np.ndarray], ...]:
    """连音数字多用斜体：普通数字模板加上几种斜体字体。用冻结的（resources/glyphs.npz），没有才用本机字体现画。"""
    from video2score.glyphs import frozen_bank

    return frozen_bank("tuplet") or _tuplet_bank_from_fonts()


def _tuplet_bank_from_fonts() -> tuple[tuple[str, np.ndarray], ...]:
    from video2score.ocr import _FONT_DIRS
    from video2score.tab import _render, _tab_bank_from_fonts

    bank = [item for item in _tab_bank_from_fonts() if item[0].isdigit()]
    for name in ("timesi.ttf", "georgiai.ttf", "cambriai.ttf", "BOOKOSI.TTF", "constani.ttf", "DejaVuSerif-Italic.ttf"):
        for directory in _FONT_DIRS:
            path = directory / name
            if path.is_file():
                bank.extend(item for item in _render(path, "0123456789") if item[0].isdigit())
                break
    return tuple(bank)


def stem_event(
    image: np.ndarray,
    staff: Staff5,
    bars: tuple[int, int],
    x: float,
    limit: tuple[float, float] | None = None,
) -> Event | None:
    """TAB 上有音、五线谱上没认出符头时（X 形的闷音符头、被背景盖住的符头），看同一位置有没有符干。

    有就按符干两头数到的符杠、符尾定时值（不知道符头在哪头，两头都数，取多的）。
    """
    spacing = staff.spacing
    top = int(max(0, staff.top - 4.0 * spacing))
    bottom = int(min(image.shape[0], staff.bottom + 4.0 * spacing))
    if limit is not None:
        top = max(top, int(limit[0]))
        bottom = min(bottom, int(limit[1]))
    ink = (255 - image[top:bottom]) >= 110
    no_lines = _without_staff_lines(ink, [line - top for line in staff.lines], spacing)
    stems = [stem for stem in _stems(no_lines, spacing, *bars) if abs(stem[0] - x) <= 0.8 * spacing]
    if not stems:
        return None
    stem_x, y0, y1 = min(stems, key=lambda stem: abs(stem[0] - x))
    up, down = _count_beams(ink, stem_x, y0, True, spacing), _count_beams(ink, stem_x, y1, False, spacing)
    beams = max(up, down)
    # 符杠在哪一头，符头就在另一头（X 形闷音符头的高度要用来推品位）。X 的符干接在它的尖上，离中心还有半个线距。
    reach = 0.5 * spacing
    if up > down:
        head_y, stem_up = float(y1 + top + reach), True
    elif down > up:
        head_y, stem_up = float(y0 + top - reach), False
    else:
        # 没有符杠符尾（四分音符）：X 符头那头，符干两侧的墨比另一头多得多。
        def side_ink(end: int) -> int:
            box = ink[max(0, end - int(spacing)) : end + int(spacing), max(0, stem_x - int(1.2 * spacing)) : stem_x + int(1.2 * spacing) + 1].copy()
            box[:, max(0, int(1.2 * spacing) - 1) : int(1.2 * spacing) + 2] = False
            return int(box.sum())
        low, high = side_ink(y1), side_ink(y0)
        if low > 2 * high and low >= 0.5 * spacing * spacing:
            head_y, stem_up = float(y1 + top + reach), True
        elif high > 2 * low and high >= 0.5 * spacing * spacing:
            head_y, stem_up = float(y0 + top - reach), False
        else:
            head_y, stem_up = None, None
    if stem_up is not None:
        # 符干常顺着 X 的一条斜臂多走一截，"末端 + 半个线距"会偏低（偏高）小半格：能量到 X 本身就用它上下两端的中点。
        measured = _x_head_center(no_lines, stem_x, y1 if stem_up else y0, stem_up, spacing)
        if measured is not None:
            head_y = measured + top
    return Event(x=float(stem_x), duration=BASE.get(min(beams, 3), Fraction(1, 4)), beams=beams, head_y=head_y, stem_up=stem_up)


def _x_head_center(no_lines: np.ndarray, stem_x: int, end: int, stem_up: bool, spacing: float) -> float | None:
    """符干那头的 X 形符头（谱线已抹掉）：符干朝上时符头在符干左边，朝下时在右边。在末端上下一个多线距里
    找挨着符干的那块墨，高度像一个符头（0.6~1.4 线距）就返回它上下两端的中点。"""
    width = int(round(1.3 * spacing))
    left, right = (max(0, stem_x - width), max(0, stem_x - 1)) if stem_up else (stem_x + 2, stem_x + width + 1)
    y0, y1 = max(0, int(end - 1.5 * spacing)), min(no_lines.shape[0], int(end + 1.5 * spacing) + 1)
    rows = np.flatnonzero(no_lines[y0:y1, left:right].any(axis=1))
    if not len(rows):
        return None
    # 只要离末端最近的那段连续的墨（谱线抹掉留下的一两行空隙不算断开）。
    gap = max(2, int(round(0.2 * spacing)))
    runs = np.split(rows, np.flatnonzero(np.diff(rows) > gap) + 1)
    run = min(runs, key=lambda r: 0 if r[0] <= end - y0 <= r[-1] else min(abs(r[0] - (end - y0)), abs(r[-1] - (end - y0))))
    height = run[-1] - run[0] + 1
    if not 0.6 * spacing <= height <= 1.4 * spacing:
        return None
    return float(y0 + (run[0] + run[-1]) / 2)


def whole_event(
    image: np.ndarray,
    staff: Staff5,
    x: float,
    limit: tuple[float, float] | None = None,
) -> Event | None:
    """TAB 上有音、同一位置也没有符干时，找没有符干的空心椭圆：全音符。

    全音符两侧粗、中间有洞，谱线常从洞里穿过，按洞找不到；和 _hollow_heads_on_stems 一样按椭圆环量。
    实心符头环里是满的，只有 0.6~0.7 分。
    """
    spacing = staff.spacing
    top = int(max(0, staff.top - 3.0 * spacing))
    bottom = int(min(image.shape[0], staff.bottom + 3.0 * spacing))
    if limit is not None:
        top = max(top, int(limit[0]))
        bottom = min(bottom, int(limit[1]))
    if bottom <= top:
        return None
    ink = (255 - image[top:bottom]) >= 110
    # 环里只算接近纯黑的墨：谱带压在视频画面上时，全音符的空心里透出来的是中灰的背景，符头本身是黑的。
    black = (255 - image[top:bottom]) >= 170
    grown = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    line_rows = _line_rows(ink.shape[0], [line - top for line in staff.lines], spacing)
    best: tuple[float, float, float] | None = None
    # 符头中心落在线上或间里：每半个线距一档，上下各留两条加线。TAB 数字和符头可能错开大半个线距。
    for step in range(-6, 15):
        cy = staff.top + 0.5 * step * spacing - top
        for dx in np.arange(-0.9, 0.91, 0.1):
            score = _ring_score(ink, grown, line_rows, x + dx * spacing, cy, spacing, black)
            if score is not None and (best is None or score > best[0]):
                best = (score, x + dx * spacing, cy)
    if best is None or best[0] < 0.85:
        return None
    _score, head_x, head_y = best
    dots = _dots(ink, head_x, head_y, spacing)
    duration = Fraction(1) * (Fraction(3, 2) if dots == 1 else Fraction(7, 4) if dots == 2 else 1)
    return Event(x=float(head_x), duration=duration, dots=dots, head_y=float(head_y + top), hollow=True)


def fermatas(image: np.ndarray, staff: Staff5, bars: tuple[int, int]) -> list[float]:
    """五线谱上方的延长记号：一道拱形（宽 1.5~2.8、高 0.8~1.6 个线距、里面是空的）底下正中一个小圆点。返回横坐标。"""
    spacing = staff.spacing
    y0 = max(0, int(staff.top - 3.0 * spacing))
    y1 = max(y0, int(staff.top - 0.1 * spacing))
    region = (image[y0:y1, bars[0] : bars[-1]] < 128).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
    boxes = [tuple(int(value) for value in stats[index]) for index in range(1, count)]
    found: list[float] = []
    for x, y, w, h, area in boxes:
        if not (1.5 * spacing <= w <= 2.8 * spacing and 0.8 * spacing <= h <= 1.6 * spacing) or area > 0.45 * w * h:
            continue
        for dx, dy, dw, dh, _darea in boxes:
            centre = dx + dw / 2.0
            if (
                0.15 * spacing <= max(dw, dh) <= 0.45 * spacing
                and x + w / 3.0 <= centre <= x + 2 * w / 3.0
                and y + h / 3.0 <= dy + dh / 2.0 <= y + h + 0.2 * spacing
            ):
                found.append(bars[0] + x + w / 2.0)
                break
    return found
