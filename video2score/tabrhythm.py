"""读纯 TAB（Guitar Pro 那种）的节奏：记号都画在 TAB 下面。

GP 的写法：
- 每个音从数字下方拉一根符干，所有符干的下端收在 TAB 下面同一条基线上。
- 符杠、符尾画在基线那头：0 道是四分，1 道八分，2 道十六分。
- 二分音符的符干短（从 TAB 下面一个半线距处才开始），全音符没有符干。
- 附点是基线上方、符干右边的小圆点。
- 延音的后半个音不写数字，只有一根从 TAB 下沿开始的符干。
- 休止符画在 TAB 里面：八分、十六分休止像带圆头的 7，四分休止是竖着的锯齿。
- 连音：基线下方一道括号，中间写 3（或 5、6），括号底下的音按比例缩短。
- 拍号：小节开头上下叠着的两个粗体数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import cv2
import numpy as np

from video2score.numbers import _fit_glyph, _glyph_corr
from video2score.ocr import _FONT_DIRS
from video2score.rhythm import BASE, Event
from video2score.tab import Bank, Onset, TabStaff, _render, classify, tab_bank

# 常见的拍号：分子一位数，分母 2、4、8。别的读数多半是背景干扰。
_NUMERATORS = (2, 3, 4, 5, 6, 7, 9)
_DENOMINATORS = (2, 4, 8)


class Meter(Fraction):
    """拍号：值、比较、加减都和分数一样，另外记着印出来的分子分母（6/8 不约成 3/4）。"""

    def __new__(cls, beats: int, beat_type: int) -> "Meter":
        self = super().__new__(cls, beats, beat_type)
        self.beats, self.beat_type = beats, beat_type
        return self

    def __reduce__(self):
        return (Meter, (self.beats, self.beat_type))

    def __repr__(self) -> str:
        return f"Meter({self.beats}, {self.beat_type})"


@dataclass
class TabRhythm:
    events: list[Event]
    # 小节开头印的拍号（没有就是 None）。印了但没读出来时 signature_seen 为真。
    meter: Fraction | None = None
    signature_seen: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class _Stem:
    x: int
    top: int
    bottom: int


def read_tab_rhythm(
    image: np.ndarray,
    staff: TabStaff,
    bars: tuple[int, int],
    groups: list[Onset],
) -> TabRhythm:
    """一小节纯 TAB：符干配上 TAB 的发声组，数符杠定时值，再加上休止符和连音。

    image 是白底黑字的灰度小节图。返回的音符事件 notes 是那一组的品位；
    notes 为空、又不是休止的事件是延音的后半个音（接前一个音）。
    """
    spacing = staff.spacing
    ink = ((255 - image) >= 110).astype(np.uint8)
    issues: list[str] = []
    meter, signature = time_signature(ink, staff, bars)
    if signature is not None:
        # 拍号的两个粗体数字会被当成品位读出来，去掉。
        groups = [group for group in groups if not signature[0] - 0.3 * spacing <= group.x <= signature[1] + 0.3 * spacing]
    stems = _stems(ink, staff, bars)
    # 四分休止（竖着的锯齿，跨三根弦）会被抹线切成几截，各自认成 1、11。先不看数字找它，和它重叠的"数字"去掉。
    for rest in _rests(ink, staff, bars, [], stems, signature):
        if rest.duration == Fraction(1, 4):
            groups = [group for group in groups if abs(group.x - rest.x) > 0.6 * spacing]
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(round(0.5 * spacing))), 1))
    )
    events: list[Event] = []
    remaining = list(groups)
    for stem in stems:
        match = min(remaining, key=lambda group: abs(group.x - stem.x), default=None)
        if match is not None and abs(match.x - stem.x) > 0.7 * spacing:
            match = None
        if match is not None:
            remaining.remove(match)
        beams = _count_beams(horizontal, stem, spacing) or _count_flags(ink, horizontal, stem, spacing)
        if beams:
            base = BASE[min(beams, 3)]
        else:
            # 二分音符的符干短：从 TAB 下面一个多线距处才开始。
            base = Fraction(1, 2) if stem.top > staff.bottom + 1.15 * spacing else Fraction(1, 4)
        dots = _dots(ink, horizontal, stem, spacing)
        duration = base * (Fraction(3, 2) if dots == 1 else Fraction(7, 4) if dots >= 2 else 1)
        notes = list(match.notes) if match is not None else []
        if match is None and stem.top < staff.bottom - 0.3 * spacing:
            # 符干从 TAB 里面伸出来，上面应该有数字，只是没读出来。
            issues.append(f"x={stem.x} 有符干、TAB 数字没读到")
        events.append(Event(x=float(stem.x), duration=duration, dots=dots, beams=beams, notes=notes))
    for group in remaining:
        if not stems:
            # 没有符干的音：全音符。
            events.append(Event(x=group.x, duration=Fraction(1, 1), notes=list(group.notes)))
            continue
        issues.append(f"x={group.x:.0f} TAB 有音、没找到符干：" + "+".join(f"{n.string}弦{n.fret}品" for n in group.notes))
    events.extend(_rests(ink, staff, bars, groups, stems, signature))
    events.sort(key=lambda event: event.x)
    baseline = int(np.median([stem.bottom for stem in stems])) if stems else int(staff.bottom + 2.5 * spacing)
    _apply_tuplets(ink, events, baseline, bars, spacing)
    return TabRhythm(events=events, meter=meter, signature_seen=signature is not None, issues=issues)


def _stems(ink: np.ndarray, staff: TabStaff, bars: tuple[int, int]) -> list[_Stem]:
    """TAB 下面的竖线，下端收在同一条基线上。"""
    spacing = staff.spacing
    tall = max(5, int(round(0.6 * spacing)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall)))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(vertical, connectivity=8)
    candidates: list[_Stem] = []
    for index in range(1, count):
        x, y, width, height, _area = (int(value) for value in stats[index])
        if width > max(4, 0.35 * spacing):
            continue
        centre = x + width // 2
        if any(abs(centre - bar) < 0.6 * spacing for bar in bars):
            continue
        # 下端要在 TAB 下面一个线距以外：圈起来的数字、括号的竖边在 TAB 里面。
        if y + height < staff.bottom + 1.0 * spacing:
            continue
        candidates.append(_Stem(centre, y, y + height))
    if not candidates:
        return []
    baseline = float(np.median([stem.bottom for stem in candidates]))
    stems = [stem for stem in candidates if abs(stem.bottom - baseline) <= 0.45 * spacing]
    # 符尾会贴着符干画出一小截竖线：挨得很近的两根只留长的。
    stems.sort(key=lambda stem: stem.x)
    kept: list[_Stem] = []
    for stem in stems:
        if kept and stem.x - kept[-1].x < 0.4 * spacing:
            if stem.bottom - stem.top > kept[-1].bottom - kept[-1].top:
                kept[-1] = stem
            continue
        kept.append(stem)
    return kept


def _count_beams(horizontal: np.ndarray, stem: _Stem, spacing: float) -> int:
    """符干下端附近、左右两侧各看一列，数有几道横向的符杠。

    一道符杠约 0.32 个线距厚，两道之间空 0.15 个线距；两道连在一起时按长度拆开。
    """
    rows = slice(max(0, stem.bottom - int(round(1.6 * spacing))), min(horizontal.shape[0], stem.bottom + 2))
    offset = max(3, int(round(0.25 * spacing)))
    unit = 0.47 * spacing
    best = 0
    for column in (stem.x - offset, stem.x + offset):
        if not 0 <= column < horizontal.shape[1]:
            continue
        profile = horizontal[rows, column].astype(bool)
        count = 0
        for length in _runs(profile):
            if length < max(2, 0.15 * spacing):
                continue
            count += max(1, int(round((length + 0.15 * spacing) / unit)))
        best = max(best, count)
    return best


def _count_flags(ink: np.ndarray, horizontal: np.ndarray, stem: _Stem, spacing: float) -> int:
    """单独一个音的符尾：从符干下端斜着往右上翘的几道短笔画，和符干连在一起。

    符干右边的附点不连着符干，不算。
    """
    reach = int(round(1.2 * spacing))
    y0, y1 = max(0, stem.bottom - int(round(1.6 * spacing))), min(ink.shape[0], stem.bottom + 2)
    x0, x1 = max(0, stem.x - 1), min(ink.shape[1], stem.x + reach)
    window = (ink[y0:y1, x0:x1] & ~horizontal[y0:y1, x0:x1]).astype(np.uint8)
    # 符干那一列也要算进来，符尾是从它长出来的。
    window[:, : min(3, window.shape[1])] |= ink[y0:y1, x0 : x0 + min(3, window.shape[1])]
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
    stem_labels = set(np.unique(labels[:, 1 : min(3, window.shape[1])])) - {0}
    attached = np.isin(labels, list(stem_labels))
    column = max(3, int(round(0.3 * spacing))) + 1
    if column >= window.shape[1]:
        return 0
    runs = [length for length in _runs(attached[:, column]) if length >= 2]
    return min(len(runs), 3)


def _runs(profile: np.ndarray) -> list[int]:
    padded = np.concatenate(([0], profile.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts, ends = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
    return list(ends - starts)


def _dots(ink: np.ndarray, horizontal: np.ndarray, stem: _Stem, spacing: float) -> int:
    """符干右边、基线上方的小圆点。"""
    x0 = stem.x + max(2, int(round(0.15 * spacing)))
    x1 = stem.x + int(round(1.0 * spacing))
    y0 = stem.bottom - int(round(1.6 * spacing))
    y1 = stem.bottom + int(round(0.3 * spacing))
    patch = (ink & ~horizontal)[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if patch.size == 0:
        return 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(patch.astype(np.uint8), connectivity=8)
    dots = 0
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if x == 0 or y == 0 or x + width >= patch.shape[1] or y + height >= patch.shape[0]:
            continue
        if not 0.12 * spacing <= max(width, height) <= 0.45 * spacing:
            continue
        if 0.6 <= width / max(1, height) <= 1.6 and area >= 0.5 * width * height:
            dots += 1
    return min(dots, 2)


def _rests(
    ink: np.ndarray,
    staff: TabStaff,
    bars: tuple[int, int],
    groups: list[Onset],
    stems: list[_Stem],
    signature: tuple[int, int] | None,
) -> list[Event]:
    """TAB 里面、不挨着数字和符干的块，按大小认成休止符。"""
    spacing = staff.spacing
    y0 = max(0, int(staff.top - 0.6 * spacing))
    y1 = min(ink.shape[0], int(staff.bottom + 0.6 * spacing) + 1)
    region = ink[y0:y1].copy()
    # 抹掉弦线：只抹线那几行里的长横线。休止符被切成几截，下面按横向重叠拼回去。
    length = max(9, int(round(1.2 * spacing)))
    long_runs = cv2.morphologyEx(region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    band = np.zeros_like(region)
    reach = max(2, int(round(0.1 * spacing)))
    for line in staff.lines:
        row = int(round(line - y0))
        band[max(0, row - reach) : row + reach + 1] = 1
    region = cv2.subtract(region, long_runs & band)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
    boxes: list[list[int]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < 3:
            continue
        boxes.append([x, y, x + width, y + height, index])
    # 被弦线切开的碎块拼回去：左右重叠、上下只隔着弦线那几行。
    boxes.sort(key=lambda box: box[1])
    clusters: list[list[list[int]]] = [[box] for box in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _touching(clusters[i], clusters[j], 2 * reach + 2):
                    clusters[i].extend(clusters.pop(j))
                    merged = True
                    break
            if merged:
                break
    taken: list[tuple[float, float]] = []
    for group in groups:
        for note in group.notes:
            taken.append((note.left - 0.3 * spacing, note.right + 0.3 * spacing))
    for stem in stems:
        taken.append((stem.x - 0.3 * spacing, stem.x + 0.3 * spacing))
    if signature is not None:
        taken.append((signature[0] - 0.3 * spacing, signature[1] + 0.3 * spacing))
    left_bar, right_bar = bars
    rests: list[Event] = []
    for cluster in clusters:
        x0 = min(item[0] for item in cluster)
        x1 = max(item[2] for item in cluster)
        top = min(item[1] for item in cluster)
        bottom = max(item[3] for item in cluster)
        width, height = x1 - x0, bottom - top
        centre = (x0 + x1) / 2.0
        # TAB 谱号的三个字母贴着左边小节线。
        if centre <= left_bar + 1.2 * spacing or centre >= right_bar - 0.5 * spacing:
            continue
        if any(a <= centre <= b for a, b in taken):
            continue
        mask = np.zeros((height, width), bool)
        for item in cluster:
            index = item[4]
            mask[item[1] - top : item[3] - top, item[0] - x0 : item[2] - x0] |= (
                labels[item[1] : item[3], item[0] : item[2]] == index
            )
        fill = float(mask.mean())
        duration: Fraction | None = None
        # 四分休止：高 2~3 个线距的粗锯齿，比带细杆的十六分休止（高 1.8、填充约 0.32）填得满。
        tall_zigzag = 1.7 * spacing <= height and fill >= 0.39 and width <= 0.8 * spacing
        if (2.2 * spacing <= height <= 3.6 * spacing or tall_zigzag) and 0.3 * spacing <= width <= 1.2 * spacing:
            duration = Fraction(1, 4)
        elif 0.85 * spacing <= height < 2.2 * spacing and 0.45 * spacing <= width <= 1.2 * spacing and fill < 0.6:
            # 八分休止高约 1.1 个线距，十六分 1.4~1.8（下面的杆被弦线切出一截）。
            duration = BASE[1] if height < 1.35 * spacing else BASE[2]
        if duration is not None:
            dots = _rest_dots(ink, x1, top + y0, spacing, taken)
            duration *= Fraction(3, 2) if dots else 1
            rests.append(Event(x=centre, duration=duration, rest=True, dots=dots))
            continue
        if 0.5 * spacing <= width <= 1.1 * spacing and 0.2 * spacing <= height <= 0.55 * spacing and fill >= 0.75:
            # 实心小方块：二分或全休止。挂在线下是全休止，坐在线上是二分休止。
            below_line = any(0 <= top + y0 - line <= 0.2 * spacing for line in staff.lines)
            rests.append(Event(x=centre, duration=Fraction(1, 1) if below_line else Fraction(1, 2), rest=True))
    return rests


def _touching(a: list[list[int]], b: list[list[int]], gap: int) -> bool:
    ax0, ay0 = min(item[0] for item in a), min(item[1] for item in a)
    ax1, ay1 = max(item[2] for item in a), max(item[3] for item in a)
    bx0, by0 = min(item[0] for item in b), min(item[1] for item in b)
    bx1, by1 = max(item[2] for item in b), max(item[3] for item in b)
    overlap = min(ax1, bx1) - max(ax0, bx0)
    if overlap < 0.3 * min(ax1 - ax0, bx1 - bx0):
        return False
    return max(ay0, by0) - min(ay1, by1) <= gap


def _rest_dots(ink: np.ndarray, right: int, top: int, spacing: float, taken: list[tuple[float, float]]) -> int:
    """休止符右上方的小圆点：附点休止。"""
    x0, x1 = right, right + int(round(0.8 * spacing))
    y0, y1 = max(0, top - int(round(0.3 * spacing))), top + int(round(0.9 * spacing))
    patch = ink[y0:y1, x0:x1]
    if patch.size == 0:
        return 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if x == 0 or x + width >= patch.shape[1] or y + height >= patch.shape[0]:
            continue
        if any(a <= x0 + x <= b for a, b in taken):
            continue
        if 0.12 * spacing <= max(width, height) <= 0.4 * spacing and 0.6 <= width / max(1, height) <= 1.6:
            return 1
    return 0


def time_signature(
    ink: np.ndarray,
    staff,
    bars: tuple[int, int],
    reach: float = 5.0,
    tall: float = 0.4,
    start: int | None = None,
) -> tuple[Fraction | None, tuple[int, int] | None]:
    """小节开头上下叠着的两个粗体数字。返回 (拍号, 横向范围)。TAB 和五线谱都能用（staff 有 lines、spacing）。

    粗体笔画有四五个像素厚，腐蚀掉两三个像素以后谱线、符干和普通数字都没了，只剩它们、实心符头、符杠和谱号的粗段。
    按横向位置分簇，上下两半各有一段够高的粗笔画、又不比一个数字宽的那一簇才是拍号。
    tall 是每半边粗笔画至少要有的高度（线距）：五线谱上有实心符头混进来，要设高一些。
    start 是行首谱号、调号结束的地方（从那里往后找，四个升号后面的拍号也够得着）。
    """
    spacing = staff.spacing
    left_bar = bars[0] if start is None else max(bars[0], start)
    y0 = max(0, int(staff.lines[0] - 0.8 * spacing))
    y1 = min(ink.shape[0], int(staff.lines[-1] + 0.8 * spacing) + 1)
    x0 = max(0, left_bar)
    x1 = min(ink.shape[1], int(left_bar + reach * spacing))
    region = ink[y0:y1, x0:x1]
    if region.size == 0:
        return None, None
    size = max(3, int(round(0.2 * spacing)))
    bold = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    middle = (staff.lines[0] + staff.lines[-1]) / 2.0 - y0
    # 分子、分母在中线处切开：紧挨着粗的双小节线时，两个数字会和它粘成一整块。
    row = int(round(middle))
    bold[max(0, row - 1) : row + 2] = 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bold, connectivity=8)
    pieces = sorted(
        (int(x), int(y), int(x + w), int(y + h))
        for x, y, w, h, area in stats[1:]
        # 小节线（又高又细）不算。
        if area >= 3 and x > 1 and not (w <= 0.4 * spacing and h >= 1.6 * spacing)
    )
    clusters: list[list[tuple[int, int, int, int]]] = []
    for piece in pieces:
        if clusters and piece[0] <= max(item[2] for item in clusters[-1]) + size:
            clusters[-1].append(piece)
        else:
            clusters.append([piece])
    found = None
    for cluster in clusters:
        left = min(item[0] for item in cluster)
        right = max(item[2] for item in cluster)
        upper = [item for item in cluster if (item[1] + item[3]) / 2.0 < middle]
        lower = [item for item in cluster if (item[1] + item[3]) / 2.0 >= middle]
        # 太窄的是断成几截的小节线（腐蚀后只剩一两列），拍号数字没这么细。
        if not upper or not lower or right - left > 2.2 * spacing or right - left < 0.3 * spacing:
            continue
        height = min(
            max(item[3] for item in half) - min(item[1] for item in half) for half in (upper, lower)
        )
        if height >= tall * spacing:
            found = (left, right)
            break
    if found is None:
        return None, None
    left, right = found
    # 腐蚀只留下最粗的竖笔，细的斜笔（4 的左半边）没了：左右放宽，下面每半边只取最大的一块。
    span = (x0 + max(0, left - size), x0 + min(region.shape[1], right + size))
    left = max(0, left - int(round(1.0 * spacing)))
    right = min(region.shape[1], right + int(round(0.8 * spacing)))
    columns = region[:, left:right].astype(bool)
    # 弦线：只去掉上下都不连着笔画的那几个像素，数字的笔画穿过弦线的地方留着。
    cleaned = columns.copy()
    reach = max(2, int(round(0.12 * spacing)))
    for line in staff.lines:
        row = int(round(line - y0))
        for r in range(max(0, row - 1), min(columns.shape[0], row + 2)):
            above = columns[max(0, row - reach - 1)]
            below = columns[min(columns.shape[0] - 1, row + reach + 1)]
            cleaned[r] &= above | below
    numbers = [_read_number(cleaned[: int(middle)], _NUMERATORS), _read_number(cleaned[int(middle) :], _DENOMINATORS)]
    if numbers[0] is None or numbers[1] is None:
        return None, span
    return Meter(numbers[0], numbers[1]), span


def multirest_digits(image: np.ndarray, top: float, spacing: float, bars: tuple[int, int]) -> list[np.ndarray]:
    """合并休止印在谱表上方的粗体小节数，一位一个字形（清洗后的小节图，白底黑字；top 是第一条谱线的 y）。

    谱表上方 0.3~6 个线距里，够高、不太宽、腐蚀掉细笔画后还剩三成以上墨的块是数字（细的小节号、速度记号剩不下）。
    """
    y0 = max(0, int(top - 6.0 * spacing))
    y1 = max(y0, int(top - 0.3 * spacing))
    region = (image[y0:y1, bars[0] : bars[1]] < 128).astype(np.uint8)
    if region.size == 0:
        return []
    size = max(2, int(round(0.15 * spacing)))
    bold = cv2.erode(region, np.ones((size, size), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
    digits: list[tuple[int, np.ndarray]] = []
    for index in range(1, count):
        x, y, w, h, _area = (int(value) for value in stats[index])
        piece = labels[y : y + h, x : x + w] == index
        if h < 0.8 * spacing or w > 2.5 * spacing:
            continue
        if bold[y : y + h, x : x + w][piece].sum() < 0.3 * piece.sum():
            continue
        digits.append((x, _fit_glyph(piece)))
    return [canvas for _x, canvas in sorted(digits, key=lambda item: item[0])]


def read_multirest(canvases: list[np.ndarray], learned: dict[str, np.ndarray] | None = None) -> int | None:
    """按字形读出合并休止的小节数。learned 是这首曲子里已知小节数的合并休止学来的字形（按数字取平均），优先用。"""
    if not canvases:
        return None
    text = ""
    for canvas in canvases:
        # 学来的字形只有这首曲子里出现过的几个数字：够像才用，不然别的数字会被硬认成它们。
        label, score = _best(canvas, list((learned or {}).items()))
        if score < 0.8:
            label, score = _best(canvas, list(_signature_bank()))
        if score < 0.4:
            return None
        text += label
    value = int(text)
    return value if 2 <= value <= 200 else None


def _best(canvas: np.ndarray, bank: list[tuple[str, np.ndarray]]) -> tuple[str, float]:
    best = ("", -1.0)
    for label, template in bank:
        score = _glyph_corr(canvas, template)
        if score > best[1]:
            best = (label, score)
    return best


def _read_number(mask: np.ndarray, allowed) -> int | None:
    """一个粗体数字（拍号只用一位数：2、3、4、6、8……），只在允许的数里挑。"""
    # 数字是最大的一块；弦线残段、旁边的字母碎块不要。抹线留下的一两行空隙先竖着补上再找。
    # 6 右上角的钩会和身子断开：再试一次"最大的一块加上和它差不多大的块"，取更像的那个。
    joined = cv2.dilate(mask.astype(np.uint8), np.ones((3, 1), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if count < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    pieces = [index + 1 for index, area in enumerate(areas) if area >= 0.15 * areas.max()]
    wanted = {str(value) for value in allowed if value < 10}
    bank = tuple(item for item in _signature_bank() if item[0] in wanted)
    best: tuple[str, float] | None = None
    for keep in ([largest], pieces):
        found = _best_digit(mask & np.isin(labels, keep), bank)
        if found is not None and (best is None or found[1] > best[1]):
            best = found
        if len(pieces) == 1:
            break
    if best is None or best[1] < 0.4:
        return None
    return int(best[0])


def _best_digit(mask: np.ndarray, bank) -> tuple[str, float] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or not bank:
        return None
    # 只在几个允许的数里挑，直接取相关值最高的（倒角裁决是为细字体的 3、8 准备的，粗体字上反而帮倒忙）。
    canvas = _fit_glyph(mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1])
    scores: dict[str, float] = {}
    for label, template in bank:
        scores[label] = max(scores.get(label, -1.0), _glyph_corr(canvas, template))
    return max(scores.items(), key=lambda item: item[1])


@lru_cache(maxsize=1)
def _signature_bank() -> Bank:
    """拍号用粗体衬线字：普通模板加上几种粗体字体。用冻结的（resources/glyphs.npz），没有才用本机字体现画。"""
    from video2score.glyphs import frozen_bank

    return frozen_bank("signature") or _signature_bank_from_fonts()


def _signature_bank_from_fonts() -> Bank:
    from video2score.tab import _tab_bank_from_fonts

    bank = [item for item in _tab_bank_from_fonts() if item[0].isdigit()]
    for name in ("timesbd.ttf", "georgiab.ttf", "cambriab.ttf", "BOOKOSB.TTF", "constanb.ttf", "DejaVuSerif-Bold.ttf"):
        for directory in _FONT_DIRS:
            path = directory / name
            if path.is_file():
                bank.extend(item for item in _render(path, "0123456789") if item[0].isdigit())
                break
    return tuple(bank)


def _apply_tuplets(ink: np.ndarray, events: list[Event], baseline: int, bars: tuple[int, int], spacing: float) -> None:
    """基线下方的连音括号：一道细横线，中间断开写着数字。括号下面的音按比例缩短。"""
    y0 = baseline + int(round(0.3 * spacing))
    y1 = min(ink.shape[0], baseline + int(round(2.6 * spacing)))
    if y1 <= y0:
        return
    region = ink[y0:y1]
    thin = cv2.morphologyEx(
        region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(round(0.8 * spacing))), 1))
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(thin, connectivity=8)
    segments = sorted(
        (int(x), int(y), int(x + w))
        for x, y, w, h, _a in stats[1:]
        if h <= max(3, 0.25 * spacing)
    )
    used = [False] * len(segments)
    for i, (ax0, ay, ax1) in enumerate(segments):
        if used[i]:
            continue
        for j in range(i + 1, len(segments)):
            bx0, by, bx1 = segments[j]
            gap = bx0 - ax1
            if used[j] or abs(by - ay) > 2 or not 0.4 * spacing <= gap <= 2.5 * spacing:
                continue
            number = _tuplet_number(region, ax1, bx0, ay, spacing)
            if number is None:
                continue
            used[i] = used[j] = True
            _scale(events, ax0 - 0.4 * spacing, bx1 + 0.4 * spacing, number)
            break


def _tuplet_number(region: np.ndarray, x0: int, x1: int, y: int, spacing: float) -> int | None:
    """括号断口里的数字，认不出就当 3。断口里没有像数字的墨，就不是连音括号。"""
    top = max(0, y - int(round(0.8 * spacing)))
    bottom = min(region.shape[0], y + int(round(0.8 * spacing)))
    patch = region[top:bottom, x0:x1].astype(bool)
    rows = np.flatnonzero(patch.any(axis=1))
    cols = np.flatnonzero(patch.any(axis=0))
    if rows.size == 0 or rows[-1] - rows[0] + 1 < 0.45 * spacing:
        return None
    crop = patch[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    digits = tuple(item for item in tab_bank() if item[0] in "35679")
    label, score, _margin = classify(_fit_glyph(crop), digits, narrow=False)
    if label is None or score < 0.45:
        return 3
    return int(label)


def _scale(events: list[Event], x0: float, x1: float, number: int) -> None:
    """n 连音占 m 个正常音的时间，m 是小于 n 的最大的 2 的幂（3 连音占 2 个，5、6 连音占 4 个）。"""
    normal = 1
    while normal * 2 < number:
        normal *= 2
    ratio = Fraction(normal, number)
    inside = [event for event in events if x0 <= event.x <= x1 and event.tuplet is None]
    for event in inside:
        event.tuplet = ratio
        event.duration *= ratio
    if inside:
        inside[0].tuplet_mark = "start"
        inside[-1].tuplet_mark = "stop" if len(inside) > 1 else "single"
