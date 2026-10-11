"""找谱行：等距长横线、谱类、横向起止和小节线。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Staff:
    """一组等距谱线。lines 是每条线的 y，按从上到下。"""

    lines: tuple[float, ...]
    spacing: float
    left: int
    right: int
    # 五弦贝斯的 TAB 也是 5 条线：和五线谱配成一行时由 find_systems 标出来。
    tab_hint: bool = False

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def bottom(self) -> float:
        return self.lines[-1]

    @property
    def count(self) -> int:
        return len(self.lines)

    @property
    def kind(self) -> str:
        """五条线是五线谱（五弦 TAB 另外标出），四条、六条、七条是 TAB。"""
        return "staff" if self.count == 5 and not self.tab_hint else "tab"


@dataclass(frozen=True)
class System:
    """一行谱。五线谱和紧贴着的 TAB 算同一行，小节线按 TAB 那组切。"""

    staves: tuple[Staff, ...]
    bars: tuple[int, ...]

    @property
    def top(self) -> float:
        return min(staff.top for staff in self.staves)

    @property
    def bottom(self) -> float:
        return max(staff.bottom for staff in self.staves)

    @property
    def spacing(self) -> float:
        return min(staff.spacing for staff in self.staves)

    @property
    def left(self) -> int:
        return min(staff.left for staff in self.staves)

    @property
    def right(self) -> int:
        return max(staff.right for staff in self.staves)

    @property
    def kind(self) -> str:
        """staff、tab 或 staff+tab。"""
        kinds = sorted({staff.kind for staff in self.staves}, key=("staff", "tab").index)
        return "+".join(kinds)

    @property
    def layout(self) -> tuple[tuple[int, float], ...]:
        """每组谱线的线数和相对第一条线的偏移（按线距计）。同一首谱的行都一样。"""
        base = self.top
        spacing = self.spacing
        return tuple(
            (staff.count, round((staff.top - base) / spacing, 1), staff.kind) for staff in self.staves
        )


def find_staves(gray: np.ndarray) -> list[Staff]:
    """先收六条，再收五条、四条等距长线。"""
    centers = line_centers(gray)
    if len(centers) < 4:
        return []
    height = gray.shape[0]
    used = [False] * len(centers)
    found: list[tuple[float, ...]] = []
    thin = _thin_dark(gray)
    # 每条线这一行（上下各多看一行）里细横线占多少、最长的一段有多长。
    cover = [_row_extent(thin, y) for y in centers]

    def accept(count: int) -> None:
        candidates: list[tuple[bool, float, int]] = []
        for start in range(len(centers) - count + 1):
            if any(used[start : start + count]):
                continue
            group = centers[start : start + count]
            gaps = [group[index + 1] - group[index] for index in range(count - 1)]
            median = sum(gaps) / len(gaps)
            if median < 4 or median > max(80.0, height * 0.08):
                continue
            if max(abs(gap - median) for gap in gaps) > max(2.0, median * 0.30):
                continue
            # 一头是一排加线的：六条的不要（会把五线谱吃掉，当成六弦 TAB），五条、四条的排到后面，
            # 让错开一条、不带这排加线的那组先挑（真的弦线被品位数字截得很碎时，没有别的组和它抢，照样收）。
            ledger = _ledger_end(cover[start : start + count], median)
            if ledger and count == 6:
                continue
            tightness = sum((gap - median) ** 2 for gap in gaps)
            candidates.append((ledger, tightness, start))
        candidates.sort()
        for _ledger, _tightness, start in candidates:
            if any(used[start : start + count]):
                continue
            for index in range(start, start + count):
                used[index] = True
            found.append(tuple(centers[start : start + count]))

    accept(6)
    accept(5)
    accept(4)
    found.sort(key=lambda lines: lines[0])
    staves: list[Staff] = []
    for lines in found:
        spacing = (lines[-1] - lines[0]) / (len(lines) - 1)
        left, right = _line_extent(thin, lines, spacing)
        staves.append(Staff(lines=lines, spacing=spacing, left=left, right=right))
    return staves


def _row_extent(thin: np.ndarray, y: float) -> tuple[float, int]:
    """这一行细横线占的比例，和最长的一段（隔一两个像素的缺口不算断）。"""
    row = int(round(y))
    line = thin[max(0, row - 1) : row + 2].any(axis=0).astype(np.uint8)
    line = cv2.morphologyEx(line[None, :], cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))[0]
    edges = np.diff(np.concatenate(([0], line.astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(line.mean()), int(lengths.max()) if lengths.size else 0


def _ledger_end(cover: list[tuple[float, int]], spacing: float) -> bool:
    """最上或最下那条"线"像一排加线：低音区密集的音，每个符头底下一小段加线，连起来也能占一行的四成多。
    加线一段只有符头那么宽（两个线距上下），这一行也比中间几条线稀。真的弦线被一排品位数字截断时也稀、段也可能短，
    所以这只是个嫌疑，由 find_staves 决定怎么用。"""
    middle = float(np.median([fraction for fraction, _longest in cover[1:-1]]))
    return any(fraction < 0.75 * middle and longest < 2.5 * spacing for fraction, longest in (cover[0], cover[-1]))


def find_systems(rgb: np.ndarray) -> list[System]:
    """把谱线组配成谱行，并找每行的小节线。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    staves = _mark_five_string_tab(find_staves(gray))
    if not staves:
        return []
    ink = bar_ink(rgb, gray, staves)
    bars = [
        _line_coloured(rgb, gray, staff, _dark_bars(gray, staff, strong_bars(ink, staff.top, staff.bottom)))
        for staff in staves
    ]
    width = gray.shape[1]
    systems: list[System] = []
    for members in _system_groups(staves, bars):
        chosen = [staves[index] for index in members]
        tab = [index for index in members if staves[index].kind == "tab"]
        bar_index = tab[0] if tab else members[0]
        # 五线谱和 TAB 之间的空当里，小节线可以断开。每组谱线上都有线才算。
        # 只穿过五线谱的符干、只穿过 TAB 的符干都不算。
        found = [
            x
            for x in bars[bar_index]
            if all(any(abs(x - other) <= 5 for other in bars[index]) for index in members)
        ]
        if len(members) >= 2:
            # 虚线小节线（GP 的自由节拍段落）：五线谱和 TAB 上都在同一列才算，符干不会伸进 TAB。
            # 播放光标是一道彩色细竖线，压在背景上看着也是断断续续的：和实线小节线一样按色调筛掉。
            dashed = [_line_coloured(rgb, gray, staves[index], _dashed_bars(gray, staves[index])) for index in members]
            # 五线谱和 TAB 中间那段空当里也有这道细线的，是从上到下贯穿谱带的播放光标，不是虚线小节线。
            ordered = sorted((staves[index] for index in members), key=lambda staff: staff.top)
            gap_top = int(ordered[0].bottom + ordered[0].spacing)
            gap_bottom = int(ordered[-1].top - ordered[0].spacing)
            thin = _thin_dark(gray.T).T if gap_bottom > gap_top else None
            for x in dashed[0]:
                if not all(any(abs(x - other) <= 3 for other in column) for column in dashed[1:]) or not all(abs(x - bar) > 8 for bar in found):
                    continue
                if thin is not None and float(thin[gap_top:gap_bottom, max(0, x - 1) : x + 2].any(axis=1).mean()) > 0.3:
                    continue
                found.append(x)
            found.sort()
        system = System(staves=tuple(chosen), bars=tuple(found))
        systems.append(_with_line_ends(system, width))
    return systems


def _dashed_bars(gray: np.ndarray, staff: Staff) -> list[int]:
    """贯穿这组谱线的细虚线：只看谱线之间的那些行（谱线行左右都是黑的，量不出竖线），
    墨占两成半到八成、断成三段以上（实线的小节线、符干都是一整段）。"""
    top, bottom = int(round(staff.top)), int(round(staff.bottom))
    if bottom - top < 8:
        return []
    rows = [row for row in range(top, bottom + 1) if all(abs(row - line) > 2 for line in staff.lines)]
    if not rows:
        return []
    thin = _thin_dark(gray.T).T & (gray < 160)
    band = thin[rows, staff.left : staff.right]
    fraction = band.mean(axis=0)
    starts = np.diff(np.vstack([np.zeros((1, band.shape[1]), bool), band]).astype(np.int8), axis=0) == 1
    runs = starts.sum(axis=0)
    columns = [index for index in range(band.shape[1]) if 0.25 <= fraction[index] <= 0.8 and runs[index] >= 3]
    groups: list[list[int]] = []
    for column in columns:
        if groups and column - groups[-1][-1] <= 2:
            groups[-1].append(column)
        else:
            groups.append([column])
    # 每个间里各量一遍：虚线在每个间里都是断断续续的；符干、TAB 上的 1 在它占的那一两个间里是满的、别的间里没有。
    spaces = [
        [index for index, row in enumerate(rows) if upper < row < lower]
        for upper, lower in zip(staff.lines, staff.lines[1:])
    ]
    found = []
    for group in groups:
        if len(group) > 4:
            continue
        column = band[:, group].any(axis=1)
        partial = sum(1 for space in spaces if space and 0.15 <= float(column[space].mean()) <= 0.85)
        if partial * 3 < 2 * len([space for space in spaces if space]):
            continue
        # 虚线贯穿每个间：哪个间整个是满的（带符尾的符干）或一点没有（TAB 上的 1 只占一两个间）都不是。
        if any(space and not 0.05 < float(column[space].mean()) < 0.95 for space in spaces):
            continue
        # 虚线小节线是孤零零的一道细线：左右隔几个像素就是空的。谱号、TAB 谱号字母（T、A、B 的竖笔）旁边都有墨。
        side = [column for offset in range(3, 7) for column in (group[0] - offset, group[-1] + offset) if 0 <= column < band.shape[1]]
        if side and float(band[:, side].mean()) > 0.08:
            continue
        found.append(staff.left + int(round(sum(group) / len(group))))
    return found


def _mark_five_string_tab(staves: list[Staff]) -> list[Staff]:
    """上下紧挨着两组 5 条线、下面那组线距明显更大：下面那组是五弦贝斯的 TAB（GP 的 TAB 线距比五线谱大一半）。"""
    from dataclasses import replace

    marked = list(staves)
    ordered = sorted(range(len(marked)), key=lambda index: marked[index].top)
    for upper, lower in zip(ordered, ordered[1:]):
        a, b = marked[upper], marked[lower]
        if a.count != 5 or b.count != 5 or a.tab_hint:
            continue
        close = b.top - a.bottom <= 8.0 * a.spacing
        if close and b.spacing >= 1.15 * a.spacing:
            marked[lower] = replace(b, tab_hint=True)
    return marked


def _dark_bars(gray: np.ndarray, staff: Staff, bars: list[int]) -> list[int]:
    """小节线和谱线一样黑。播放光标、插画描边虽然也细，但明显更浅。"""
    if not bars:
        return bars
    y0 = int(round(staff.top))
    y1 = int(round(staff.bottom)) + 1
    line_rows = [int(round(y)) for y in staff.lines]
    line_level = float(np.median(gray[line_rows, staff.left : staff.right]))
    paper = float(np.percentile(gray[y0:y1, staff.left : staff.right], 75))
    limit = (line_level + paper) / 2.0
    between = [row for row in range(y0, y1) if all(abs(row - line) > 2 for line in line_rows)]
    if not between:
        return bars
    kept: list[int] = []
    for x in bars:
        # 双小节线合并后，中心落在两条线之间的空隙上。取附近最黑的一列。
        window = gray[between, max(0, x - 6) : x + 7]
        if window.size and float(np.median(window, axis=0).min()) <= limit:
            kept.append(x)
    return kept


def _line_coloured(rgb: np.ndarray, gray: np.ndarray, staff: Staff, bars: list[int]) -> list[int]:
    """小节线和谱线一个色调。播放光标（比如压在紫色高亮上的淡绿竖线）色差不大、也够深，但色调不对。

    和紧挨着它的谱线比：两边在同一层半透明高亮下面，染的色一样。只比色调（减掉三通道平均），
    不比深浅：GP 的谱线是灰的，小节线是黑的。
    """
    if not bars:
        return bars
    y0 = int(round(staff.top))
    y1 = int(round(staff.bottom)) + 1
    line_rows = [int(round(y)) for y in staff.lines]
    between = [row for row in range(y0, y1) if all(abs(row - line) > 2 for line in line_rows)]
    if not between:
        return bars
    # 只取用到的那几个像素再转浮点：整帧转一次比整个函数都慢（每帧每行谱都调）。
    colour = rgb
    width = gray.shape[1]
    kept: list[int] = []
    for x in bars:
        lo, hi = max(0, x - 6), min(width, x + 7)
        if hi <= lo:
            continue
        # 竖线本身：附近最深的一列（双小节线合并后中心落在两条线之间），谱线之间那些行的中位色。
        column = lo + int(np.argmin(np.median(gray[between, lo:hi], axis=0)))
        bar = np.median(colour[between, column].astype(np.float32), axis=0)
        # 两侧分别比，有一侧对得上就行：行首、行尾的线外面没有谱线。
        for sides in (range(column - 12, column - 7), range(column + 8, column + 13)):
            line = _line_colour(colour, gray, line_rows, [c for c in sides if 0 <= c < width])
            if line is not None and float(np.abs(_tint(bar) - _tint(line)).max()) <= 30:
                kept.append(x)
                break
    return kept


def _tint(colour: np.ndarray) -> np.ndarray:
    """色调：三个通道各自减掉平均值。灰色是 0。"""
    return colour - colour.mean()


def _line_colour(colour: np.ndarray, gray: np.ndarray, line_rows: list[int], columns: list[int]) -> np.ndarray | None:
    """这几列上谱线的中位色。谱线只有一两像素粗：每条线在上下一像素里取最深的那个点。"""
    samples = []
    for row in line_rows:
        rows = [r for r in (row - 1, row, row + 1) if 0 <= r < gray.shape[0]]
        for c in columns:
            darkest = rows[int(np.argmin(gray[rows, c]))]
            samples.append(colour[darkest, c])
    return np.median(np.array(samples, dtype=np.float32), axis=0) if samples else None


def _with_line_ends(system: System, width: int) -> System:
    """谱线在画面里就结束时，两端就是这一行的边界，和小节线一样用。"""
    bars = list(system.bars)
    reach = max(6.0, system.spacing)
    ends = []
    if system.left > 3:
        ends.append(system.left)
    if system.right < width - 3:
        ends.append(system.right - 1)
    for x in ends:
        if all(abs(x - bar) > reach for bar in bars):
            bars.append(x)
    return System(staves=system.staves, bars=tuple(sorted(bars)))


def bar_ink(rgb: np.ndarray, gray: np.ndarray, staves: list[Staff]) -> np.ndarray:
    """可能是小节线的像素：深、不带颜色、左右两侧都更亮。

    谱带后面灰色插画的竖边只有一侧亮；播放光标和高亮框是彩色的；这些都不算。
    谱线所在的几行左右也是黑的，量不出竖线，直接当作有墨。
    """
    vertical = _thin_dark(gray.T).T | _thin_dark(gray.T, 7).T
    # 用色差而不是 HSV 饱和度：绿高亮下面的黑线饱和度很高，色差却很小。
    # 只看比左右多出来的色差：半透明的高亮框把整片（连同里面的小节线）都染了色，线和周围一样；
    # 光标是一道细的彩色竖线，比左右两侧色重得多。
    color = chroma(rgb)
    around = cv2.morphologyEx(color, cv2.MORPH_OPEN, np.ones((1, 9), np.uint8))
    ink = vertical & (gray < 140) & (cv2.subtract(color, around) < 60)
    for staff in staves:
        for y in staff.lines:
            row = int(round(y))
            band = slice(max(0, row - 2), row + 3)
            # 这几行只看亮度，竖线穿过谱线时一样是黑的。
            ink[band] |= gray[band] < 140
    return np.where(ink, np.uint8(255), np.uint8(0))


def chroma(rgb: np.ndarray) -> np.ndarray:
    """最亮通道减最暗通道。黑白灰接近 0，彩色高亮和光标很大。"""
    red, green, blue = cv2.split(rgb)
    return cv2.subtract(cv2.max(cv2.max(red, green), blue), cv2.min(cv2.min(red, green), blue))


def line_centers(gray: np.ndarray) -> list[float]:
    """两侧都更亮的细横线。色块边缘只有一侧更亮，不收。

    谱线常被音符截成短段，不能要求一整行连续。这一行里细线像素够多即可。
    """
    height, width = gray.shape
    if width < 16 or height < 16:
        return []
    fraction = _thin_dark(gray).mean(axis=1)
    rows = [index for index, value in enumerate(fraction) if value >= 0.40]
    if not rows:
        return []
    clusters: list[list[int]] = [[rows[0]]]
    for row in rows[1:]:
        if row - clusters[-1][-1] <= 3:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    centers: list[float] = []
    for cluster in clusters:
        if len(cluster) <= 6:
            centers.append(sum(cluster) / len(cluster))
    return centers


def _thin_dark(gray: np.ndarray, radius: int = 3) -> np.ndarray:
    """上下 radius 行都比它亮 12 以上的像素。"""
    gray = np.ascontiguousarray(gray)
    padded = cv2.copyMakeBorder(gray, radius, radius, 0, 0, cv2.BORDER_REPLICATE)
    surround = cv2.min(padded[: -2 * radius], padded[2 * radius :])
    return cv2.subtract(surround, gray) >= 12


def _line_extent(thin: np.ndarray, lines: tuple[float, ...], spacing: float) -> tuple[int, int]:
    """谱线在这一行里从哪一列画到哪一列。多数线上都有细线才算。"""
    width = thin.shape[1]
    votes = np.zeros(width, np.int32)
    for y in lines:
        row = int(round(y))
        votes += thin[max(0, row - 1) : row + 2].any(axis=0).astype(np.int32)
    present = votes >= max(2, int(np.ceil(len(lines) * 0.6)))
    # 符头、数字会把谱线截断几像素。闭运算补上这些缺口。
    gap = max(9, int(round(spacing * 2.5)))
    closed = cv2.morphologyEx(
        present.astype(np.uint8)[None, :],
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)),
    )[0].astype(bool)
    columns = np.flatnonzero(closed)
    if columns.size == 0:
        return 0, width
    # 取最长的连续一段，避免页边的零星横线把范围拉宽。
    breaks = np.flatnonzero(np.diff(columns) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [columns.size - 1]))
    best = int(np.argmax(columns[ends] - columns[starts]))
    return int(columns[starts[best]]), int(columns[ends[best]]) + 1


def _system_groups(staves: list[Staff], bars: list[list[int]]) -> list[list[int]]:
    """一根五线谱只配最近的一组 TAB。只剩 TAB 时不互相合并。"""
    partner: dict[int, int] = {}
    for index, staff in enumerate(staves):
        if staff.kind != "staff":
            continue
        best: int | None = None
        best_distance: float | None = None
        for other, candidate in enumerate(staves):
            if candidate.kind != "tab" or other in partner:
                continue
            if len(bars[other]) < 2:
                continue
            distance = max(0.0, max(staff.top, candidate.top) - min(staff.bottom, candidate.bottom))
            limit = max((staff.bottom - staff.top) * 2.2, staff.spacing * 8.0)
            if candidate.spacing >= 1.15 * staff.spacing and candidate.top > staff.bottom:
                # 下面线距更大的 TAB（GP 的 TAB 线距比五线谱大一半）：横向滚动谱常把它放得很远。
                limit = max(limit, staff.spacing * 12.0)
            if distance <= limit and (best_distance is None or distance < best_distance):
                best = other
                best_distance = distance
        if best is not None:
            partner[best] = index
    groups: list[list[int]] = []
    for index, staff in enumerate(staves):
        if staff.kind == "tab" and index in partner:
            continue
        tabs = [tab for tab, owner in partner.items() if owner == index]
        groups.append([index, *tabs])
    groups.sort(key=lambda members: min(staves[item].top for item in members))
    return groups


def strong_bars(ink: np.ndarray, top: float, bottom: float) -> list[int]:
    """几乎贯穿这一组谱线的细竖线。TAB 上这就是小节线，符干很少能铺满。"""
    y0 = int(round(top))
    y1 = int(round(bottom))
    if y1 - y0 < 8 or ink.shape[1] < 8:
        return []
    fraction = ink[y0 : y1 + 1].mean(axis=0) / 255.0
    columns = [index for index, value in enumerate(fraction) if value >= 0.85]
    if not columns:
        return []
    groups: list[list[int]] = [[columns[0]]]
    for column in columns[1:]:
        if column - groups[-1][-1] <= 4:
            groups[-1].append(column)
        else:
            groups.append([column])
    # 实心色块会铺满很多列。粗终止线也不过谱高的四分之一。
    widest = max(4, int((y1 - y0) * 0.25))
    centers = [
        int(round(sum(group) / len(group)))
        for group in groups
        if len(group) <= widest
    ]
    merged: list[int] = []
    for center in centers:
        if merged and center - merged[-1] <= 14:
            merged[-1] = (merged[-1] + center) // 2
            continue
        merged.append(center)
    return merged


def without_staff_lines(mask: np.ndarray) -> np.ndarray:
    """抽掉细长的谱线，避免贴着谱线的数字和整行谱粘成一块。"""
    _height, width = mask.shape
    if width < 48:
        return mask
    kernel = max(28, width // 8)
    lines = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, 1)),
    )
    thick = cv2.morphologyEx(
        lines,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, 5)),
    )
    return cv2.subtract(mask, cv2.subtract(lines, thick))


def legacy_systems(gray: np.ndarray) -> list[tuple[float, float, float, int]]:
    """(上线, 下线, 间距, 线数)。小节号读取和裁谱用这个简单形式。"""
    return [(staff.top, staff.bottom, staff.spacing, staff.count) for staff in find_staves(gray)]
