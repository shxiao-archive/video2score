"""读 TAB：压在弦线上的品位数字 → (弦, 品, 横坐标)，再按横坐标合成同时发声的一组。

输入是小节库里清洗后的小节图（白底黑字）和 measures.json 里的谱线位置。
弦从上往下数：第 1 弦是最上面那条线（音最高）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import cv2
import numpy as np

from video2score.numbers import _DIGIT_BANKS, _fit_glyph, _glyph_corr, corr_all
from video2score.ocr import _font_paths, fit_gray

# 认一个字至少要这么像，并且比第二名高出这么多。
MIN_SCORE = 0.60
MIN_MARGIN = 0.04

# 模板里除了数字，还有 TAB 谱号的字母、括号和滑音斜线：认成它们的块不当品位。
_EXTRA = ("T", "A", "B", "(", ")", "x", "X", "/", "\\")

Bank = tuple[tuple[str, np.ndarray], ...]


@dataclass
class TabStaff:
    """一组 TAB 谱线。lines 是小节图里的绝对 y，从上到下。"""

    lines: tuple[float, ...]
    spacing: float

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def bottom(self) -> float:
        return self.lines[-1]

    @property
    def strings(self) -> int:
        return len(self.lines)


@dataclass
class FretNote:
    string: int
    fret: int
    x: float
    left: int
    right: int
    score: float
    # 带括号：延音（接前一个音）。GP、MuseScore 都这样画延音后的音。
    paren: bool = False
    # 后面紧跟的滑音斜线："/" 往上滑，"\\" 往下滑。slide_next：斜线后面同一组里还有音（从它滑到那个音），
    # 否则是滑出（"/" 向上滑出，"\\" 向下滑出）。
    slide: str | None = None
    slide_next: bool = False
    # 前面紧挨的斜线（一组数字开头的斜线）：滑入，"/" 是从下滑入。
    slide_in: str | None = None
    # 后面那道斜线的横坐标（离前后哪个音近）。
    slide_x: float | None = None
    # 和下一个音之间既有斜线、又有弧线：连奏滑音，另写连线。
    slur_next: bool = False
    # 斜线上方印着 "sl."：连奏滑音（只写滑音，不写连线）。
    legato: bool = False
    # 闷音：TAB 上写 X，品位记 0。
    dead: bool = False
    # 上面搭着的弧线（击弦、勾弦）："start" 弧线往右伸，"stop" 往左伸。
    arc: str | None = None
    # 和下一个音（同一根弦）之间有一道分开画的弧线（数字上方或下方）。
    arc_next: bool = False
    # 比一般的数字小一号：装饰音。
    small: bool = False
    # 投票时正好一半的帧读到它（两帧里只有一帧）：要清洗图也读得出来才留。
    weak: bool = field(default=False, repr=False)
    # 斜线是一半对一半投出来的（两帧里只有一帧看到）：清洗图上同一个音没有斜线就去掉。
    slide_weak: bool = field(default=False, repr=False)
    canvases: list[np.ndarray] = field(repr=False, default_factory=list)


@dataclass
class Onset:
    """同一时刻发声的一组音（单音或和弦）。"""

    x: float
    notes: list[FretNote]


@dataclass
class _Glyph:
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray
    string: int = -1
    # 去掉粘在两端的弦线残段后的样子。原样认不出时再试它（括号后面常粘一段线）。
    trimmed: "_Glyph | None" = None

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@lru_cache(maxsize=1)
def tab_bank() -> Bank:
    """数字（手工模板 + 字体）以及字母、括号（字体）：用冻结的（resources/glyphs.npz），没有才用本机字体现画。"""
    from video2score.glyphs import frozen_bank

    return frozen_bank("tab") or _tab_bank_from_fonts()


def _tab_bank_from_fonts() -> Bank:
    bank: list[tuple[str, np.ndarray]] = [(str(digit), template) for digit, template in _DIGIT_BANKS]
    for path in _font_paths():
        bank.extend(_render(path, "0123456789" + "".join(_EXTRA)))
    return tuple(bank)


def _render(path: object, characters: str) -> list[tuple[str, np.ndarray]]:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(str(path), 64)
    except OSError:
        return []
    rendered: list[tuple[str, np.ndarray]] = []
    for character in characters:
        canvas = Image.new("L", (110, 110), 0)
        ImageDraw.Draw(canvas).text((20, 10), character, fill=255, font=font)
        ink = np.array(canvas) > 127
        rows = np.flatnonzero(ink.any(axis=1))
        cols = np.flatnonzero(ink.any(axis=0))
        if not rows.size:
            continue
        crop = ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        rendered.append((character, _fit_glyph(crop)))
        if character.isdigit():
            # 有的谱面数字压在弦线上、线不断开，数字里穿过一道横线（0 看起来像 θ）。
            # 线穿过的高度因字体而异：有的在中间，有的正好压着 7 的上横，所以顶、1/4、中、3/4、底各做一个。
            thickness = max(1, crop.shape[0] // 14)
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                crossed = crop.copy()
                row = min(crossed.shape[0] - thickness, int(round(fraction * (crossed.shape[0] - thickness))))
                crossed[row : row + thickness] = True
                rendered.append((character, _fit_glyph(crossed)))
    return rendered


def tab_staff(record: dict) -> TabStaff | None:
    """从 measures.json 的一条记录里取出 TAB 那组谱线（4、5、6 或 7 条中非五线谱的那组）。"""
    top = float(record["staff_top"])
    lines = [top + float(value) for value in record["lines"]]
    start = 0
    for item in record["layout"]:
        count = int(item[0])
        group = lines[start : start + count]
        start += count
        # layout 里标了类型就按标记（五弦 TAB 也是 5 条线）；旧作业没标，4、6、7 条的是 TAB。
        is_tab = item[2] == "tab" if len(item) > 2 else count in (4, 6, 7)
        if is_tab and count >= 4:
            return TabStaff(lines=tuple(group), spacing=(group[-1] - group[0]) / (count - 1))
    return None


def read_frets(
    image: np.ndarray,
    staff: TabStaff,
    bank: Bank | None = None,
    bars: tuple[int, int] | None = None,
) -> list[FretNote]:
    """找出压在弦线上的数字。image 是白底黑字的灰度小节图，bars 是左右小节线的横坐标。"""
    bank = bank or tab_bank()
    spacing = staff.spacing
    y0 = max(0, int(staff.top - 0.8 * spacing))
    y1 = min(image.shape[0], int(staff.bottom + 0.8 * spacing) + 1)
    region = (255 - image[y0:y1]).astype(np.uint8)
    lines = [line - y0 for line in staff.lines]
    glyphs = _glyphs(region, lines, spacing)
    if bars is not None:
        # 小节线抹掉后在弦线之间留下的小竖截，紧贴着小节线，会被认成 1。
        reach = 0.35 * spacing
        glyphs = [
            glyph
            for glyph in glyphs
            if not any(glyph.x0 - reach <= bar <= glyph.x1 + reach for bar in bars)
        ]
    if not glyphs:
        return []
    # 字高：只看够高的块。被弦线切成两半的斜线、碎块很多时，全体的中位数会被拉低。
    heights = np.array([glyph.height for glyph in glyphs], float)
    typical = float(np.median(heights[heights >= 0.6 * np.percentile(heights, 90)]))
    notes: list[FretNote] = []
    for string in range(staff.strings):
        row = sorted((glyph for glyph in glyphs if glyph.string == string), key=lambda glyph: glyph.x0)
        found: list[FretNote] = []
        lone: list[tuple[_Glyph, str]] = []
        for group in _groups(row, typical):
            read = _read_group(group, region, string, bank, typical)
            # 没有数字、只有一两块方向一致的斜笔（斜线常被弦线切成上下两半）：一道单独的斜线。
            slashes = [_lone_slash(glyph, region, bank, typical) for glyph in group] if not read and len(group) <= 2 else []
            if slashes and slashes[0] is not None and all(slash == slashes[0] for slash in slashes):
                box = _Glyph(min(g.x0 for g in group), min(g.y0 for g in group), max(g.x1 for g in group), max(g.y1 for g in group), np.zeros((1, 1), bool), string)
                lone.append((box, slashes[0]))
            found.extend(read)
        _attach_slashes(found, lone, typical)
        # 斜线后面的音分在另一组里（隔得稍远）：离斜线不到 1.2 个字宽，也算两个音之间的斜线。
        ordered = sorted(found, key=lambda note: note.x)
        for first, second in zip(ordered, ordered[1:]):
            if first.slide and not first.slide_next and first.slide_x is not None and second.left - first.slide_x <= 1.2 * typical:
                first.slide_next = True
            elif first.slide is None and second.slide_in is None:
                _between_slide(image, staff, first, second)
        # 这根弦上最后一个音后面一直伸到小节线的斜线：连奏滑音连到下一小节（斜线贴着弦线，找字块时被当成弦线抹掉了）。
        if ordered and bars is not None and ordered[-1].slide is None:
            _exit_slide(image, staff, ordered[-1], bars[-1])
        notes.extend(found)
    return sorted(notes, key=lambda note: (note.x, note.string))


def _flat_slash(image: np.ndarray, staff: TabStaff, string: int, x0: int, x1: int) -> str | None:
    """x0~x1 这一段里贴着弦线的一道平缓斜线："/" 或 "\\"，没有是 None。
    弦线上下 0.6 个线距内（弦线那几行除外）的墨横跨了大半段，每列墨的平均高度随横坐标一路升或一路降。"""
    spacing = staff.spacing
    y = staff.lines[string - 1]
    if x1 - x0 < 0.8 * spacing:
        return None
    ink = (255 - image) >= 110
    reach = max(2, int(round(0.1 * spacing)))
    top, bottom = max(0, int(y - 0.6 * spacing)), min(image.shape[0], int(y + 0.6 * spacing) + 1)
    window = ink[top:bottom, x0:x1].copy()
    line = int(round(y)) - top
    window[max(0, line - reach) : line + reach + 1] = False
    ys, xs = np.nonzero(window)
    columns = np.unique(xs)
    # 斜线很平，挨着弦线的那一段和弦线叠在一起被去掉了：剩下的盖住一半以上的列就行。按每列墨的平均高度量斜不斜。
    if columns.size < max(6, 0.5 * (x1 - x0)):
        return None
    heights = np.array([ys[xs == column].mean() for column in columns])
    if np.std(heights) == 0:
        return None
    slope = float(np.corrcoef(columns, heights)[0, 1])
    if abs(slope) < 0.8:
        return None
    return "/" if slope < 0 else "\\"


def _between_slide(image: np.ndarray, staff: TabStaff, first: FretNote, second: FretNote) -> None:
    """同一根弦上相邻两个音之间整段是一道很平的斜线（几乎贴着弦线，找字块时和弦线一起被抹掉）：两音之间的滑音。
    是不是连奏滑音（"sl."、弧线）后面另看。"""
    slash = _flat_slash(image, staff, first.string, first.right + 2, second.left - 2)
    if slash is None:
        return
    first.slide = slash
    first.slide_x = (first.right + second.left) / 2.0
    first.slide_next = True
    first.legato = True


def _exit_slide(image: np.ndarray, staff: TabStaff, note: FretNote, right_bar: int) -> None:
    """最后一个音右边到小节线之间是一道斜线：伸向下一小节的连奏滑音；外侧那条带里还有一道细墨从这个音附近一直伸到小节线，再写连线。"""
    spacing = staff.spacing
    y = staff.lines[note.string - 1]
    slash = _flat_slash(image, staff, note.string, note.right + 2, right_bar - 2)
    if slash is None:
        return
    ink = (255 - image) >= 110
    reach = max(2, int(round(0.1 * spacing)))
    note.slide = slash
    note.slide_x = (note.right + right_bar) / 2.0
    note.legato = True
    # 外侧的弧线：弦线上方或下方 0.6~1.5 个线距的带里，一道细墨从音附近一直伸到小节线跟前。
    for side in (-1, 1):
        b0 = max(0, int(y - 1.5 * spacing if side < 0 else y + 0.6 * spacing))
        b1 = min(image.shape[0], int(y - 0.6 * spacing if side < 0 else y + 1.5 * spacing))
        band = ink[b0:b1, note.left : right_bar - 1].astype(np.uint8)
        if band.size == 0:
            continue
        # 带里可能有别的弦线（音在第 4 弦时上方那条带里就是第 3 弦）：整块都在弦线那几行上的只是弦线，不是弧线。
        line_rows = {int(round(other)) - b0 + offset for other in staff.lines for offset in range(-reach, reach + 1)}
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(band, connectivity=8)
        for index in range(1, count):
            x, y_top, w, h, area = (int(value) for value in stats[index])
            if all(y_top + row in line_rows for row in range(h)):
                continue
            if x <= 0.8 * spacing and x + w >= band.shape[1] - 0.5 * spacing and h <= 1.0 * spacing and area <= 0.3 * spacing * w:
                note.slur_next = True
                return


def _lone_slash(glyph: _Glyph, region: np.ndarray, bank: Bank, typical: float) -> str | None:
    """离数字稍远、自成一组的斜线（一笔、和数字差不多高、认不成有把握的数字）。
    斜线穿过弦线的地方留着一整行弦线，看是不是一笔时先去掉这种通栏的行。"""
    if not 0.4 * typical <= glyph.height <= 1.2 * typical or glyph.width > 1.3 * typical:
        return None
    keep = glyph.mask.sum(axis=1) < 0.8 * glyph.width
    stroke = replace(glyph, mask=glyph.mask[keep]) if keep.any() and not keep.all() else glyph
    if not _single_stroke(stroke):
        return None
    label, score, _canvas = _read_glyph(glyph, region, bank)
    if label is not None and label.isdigit() and score >= 0.7:
        return None
    return _slash_direction(glyph)


def _attach_slashes(notes: list[FretNote], lone: list[tuple[_Glyph, str]], typical: float) -> None:
    """自成一组的斜线挂到同一根弦上紧挨着的音：在音右边是滑出，在音左边是滑入，两边都挨着是两音之间的滑音。"""
    reach = 1.2 * typical
    for glyph, slash in lone:
        before = max((note for note in notes if 0 <= glyph.x0 - note.right <= reach), key=lambda note: note.right, default=None)
        after = min((note for note in notes if 0 <= note.left - glyph.x1 <= reach), key=lambda note: note.left, default=None)
        if before is not None and before.slide is None:
            before.slide = slash
            before.slide_x = (glyph.x0 + glyph.x1) / 2.0
            before.slide_next = after is not None
        elif after is not None and after.slide_in is None:
            after.slide_in = slash


def _glyphs(region: np.ndarray, lines: list[float], spacing: float) -> list[_Glyph]:
    """弦线附近、像一个字那么大的墨迹块。"""
    ink = (region >= 110).astype(np.uint8)
    # 先抹符干和小节线：比 1.1 个线距长的竖线（数字的竖笔没这么长）。
    # 要在抹弦线之前做：这时符干还是穿过弦线的一整条；弦线抹掉后它断成几小截，就抹不干净了。
    tall = max(9, int(round(1.1 * spacing)))
    stems = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall)))
    # 竖线穿过弦线的那一点也属于弦线，留着，免得把弦线切断。
    without_stems = cv2.subtract(ink, stems)
    # 只在弦线那几行里抹掉长横线。数字的横笔若和弦线连成一片，也只丢这两三行，下面再拼回去。
    length = max(9, int(round(1.2 * spacing)))
    long_runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    band = np.zeros_like(ink)
    reach = max(2, int(round(0.1 * spacing)))
    for line in lines:
        row = int(round(line))
        band[max(0, row - reach) : row + reach + 1] = 1
    cleaned = cv2.subtract(without_stems, long_runs & band)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    pieces: list[_Glyph] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        # 小碎块先留着：抹线切下来的一小截字尾，要拼回去。
        # 圈起来的数字（GP 的二分、全音符）连着后面的延音弧，会比一个字宽不少，也先留着。
        circle_like = 1.0 * spacing <= height and width <= 1.35 * height
        if width > (1.6 if circle_like else 1.3) * spacing and height <= 1.8 * spacing:
            # 圈起来的数字后面拖着一条长延音弧（⑤⌣⌣(5)），抹线后下半圈和弧连成一长条：
            # 截下最左边一截留着（圈在左边），能和上半圈拼回去；右边的弧不要。
            keep = int(round(1.25 * spacing))
            part = labels[y : y + height, x : x + keep] == index
            rows = np.flatnonzero(part.any(axis=1))
            if rows.size:
                part = part[rows[0] : rows[-1] + 1]
                pieces.append(_Glyph(x, y + int(rows[0]), x + keep, y + int(rows[-1]) + 1, part))
            continue
        if area < 2 or width > (1.6 if circle_like else 1.3) * spacing or height > 1.8 * spacing:
            continue
        pieces.append(_Glyph(x, y, x + width, y + height, labels[y : y + height, x : x + width] == index))
    glyphs = [_open_circle(glyph, spacing, ink) for glyph in _rejoin(pieces, ink, reach, band, 1.3 * spacing)]
    glyphs = _distinct([glyph for glyph in glyphs if int(glyph.mask.sum()) >= 4])
    glyphs = [_through_line(glyph, ink, band) for glyph in glyphs]
    kept: list[_Glyph] = []
    for glyph in glyphs:
        # 太宽的块多半是数字的底边连上了旁边的弦线（12 的 2），先剪掉两端只在线上的部分。
        if glyph.width > 1.1 * spacing:
            glyph = _trim_line_stubs(glyph, lines, reach, force=True)
        glyph.trimmed = _trim_line_stubs(glyph, lines, reach)
        # 括号比数字高，放到 1.7 个线距。比一个字宽的（"0/" 这种数字粘着斜线）也留着，读的时候再拆。
        if not 0.40 * spacing <= glyph.height <= 1.7 * spacing or glyph.width > 1.6 * spacing:
            continue
        centre = (glyph.y0 + glyph.y1) / 2.0
        string = int(np.argmin([abs(centre - line) for line in lines]))
        if abs(centre - lines[string]) > 0.42 * spacing:
            continue
        glyph.string = string
        kept.append(glyph)
    return kept


def _distinct(glyphs: list[_Glyph]) -> list[_Glyph]:
    """外框几乎一样的只留一个：弦线把圈切成上下两截时，两截打开后是同一个数字。"""
    kept: list[_Glyph] = []
    for glyph in sorted(glyphs, key=lambda item: -item.width * item.height):
        area = glyph.width * glyph.height
        duplicate = False
        for other in kept:
            w = min(glyph.x1, other.x1) - max(glyph.x0, other.x0)
            h = min(glyph.y1, other.y1) - max(glyph.y0, other.y0)
            if w > 0 and h > 0 and w * h >= 0.8 * max(area, other.width * other.height):
                duplicate = True
                break
        if not duplicate:
            kept.append(glyph)
    return kept


def _through_line(glyph: _Glyph, ink: np.ndarray, band: np.ndarray) -> _Glyph:
    """字形的上沿或下沿紧挨着一条弦线时，那几行弦线很可能是字的一部分（压在线上的 7 的上横）。

    把字框延伸过去，框内的弦线像素也算进字形；模板里有带横线的版本来对它。
    """
    rows = band[:, 0].astype(bool)
    y0, y1 = glyph.y0, glyph.y1
    # 中间可能隔着一两行抗锯齿的浅色过渡，3 行以内都算紧挨着。
    above = [row for row in range(max(0, y0 - 3), y0) if rows[row]]
    if above:
        y0 = above[0]
        while y0 > 0 and rows[y0 - 1]:
            y0 -= 1
    below = [row for row in range(y1, min(len(rows), y1 + 3)) if rows[row]]
    if below:
        y1 = below[-1] + 1
        while y1 < len(rows) and rows[y1]:
            y1 += 1
    if y0 == glyph.y0 and y1 == glyph.y1:
        return glyph
    mask = np.zeros((y1 - y0, glyph.width), bool)
    mask[glyph.y0 - y0 : glyph.y0 - y0 + glyph.height] = glyph.mask
    lined = ink[y0:y1, glyph.x0 : glyph.x1].astype(bool) & rows[y0:y1, None]
    mask |= lined
    extended = _Glyph(glyph.x0, y0, glyph.x1, y1, mask, glyph.string)
    return extended


def _trim_line_stubs(glyph: _Glyph, lines: list[float], reach: int, force: bool = False) -> _Glyph:
    """字形左右两端只落在弦线那几行里的列，是粘上的一小段弦线（括号后面常有），裁掉。

    默认只修窄字形；force 时不论宽窄都修（整块比一个字还宽时）。
    """
    rows = np.arange(glyph.y0, glyph.y1)
    in_band = np.zeros(len(rows), bool)
    for line in lines:
        in_band |= np.abs(rows - line) <= reach + 1
    if in_band.all() or not in_band.any():
        return glyph
    outside = glyph.mask[~in_band].any(axis=0)
    columns = np.flatnonzero(outside)
    if columns.size == 0:
        return glyph
    left, right = int(columns[0]), int(columns[-1]) + 1
    if left == 0 and right == glyph.width:
        return glyph
    # 只修窄字形（括号、1）。4 的横笔、2 的底笔也压在线上并伸出去，不能剪。
    if not force and right - left >= 0.45 * glyph.height:
        return glyph
    mask = glyph.mask[:, left:right]
    rows_used = np.flatnonzero(mask.any(axis=1))
    top, bottom = int(rows_used[0]), int(rows_used[-1]) + 1
    return _Glyph(glyph.x0 + left, glyph.y0 + top, glyph.x0 + right, glyph.y0 + bottom, mask[top:bottom])


def _open_circle(glyph: _Glyph, spacing: float, ink: np.ndarray | None = None) -> _Glyph:
    """圈起来的品位（GP 的 ⑤）：外面一圈细环，里面一个数字。只留里面的数字。

    弦线常从圈里的数字中间穿过，抹线后数字缺了一截。给了 ink 时，数字框里的墨按原图取，
    穿过的那段弦线也留着，交给带横线的模板去认。
    """
    height, width = glyph.mask.shape
    # 圈比数字大，而且接近正圆。
    if not 1.1 * spacing <= height <= 1.8 * spacing or not 0.75 <= width / max(1, height) <= 1.3:
        return glyph
    border = max(2, int(round(0.18 * min(height, width))))
    ring = np.zeros_like(glyph.mask)
    ring[:border], ring[-border:], ring[:, :border], ring[:, -border:] = True, True, True, True
    # 四条边都有四成以上被墨盖住：是一个圈。4 的上边只有一个尖、下边只有一根竖笔，不算。
    sides = [
        glyph.mask[:border].any(axis=0).mean(),
        glyph.mask[-border:].any(axis=0).mean(),
        glyph.mask[:, :border].any(axis=1).mean(),
        glyph.mask[:, -border:].any(axis=1).mean(),
    ]
    if min(sides) < 0.4:
        return glyph
    if ink is not None:
        opened = _circle_digit(glyph, ink)
        if opened is not None and opened.height >= 0.4 * height:
            return opened
    inner = glyph.mask & ~cv2.dilate(
        (glyph.mask & ring).astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(inner.astype(np.uint8), connectivity=8)
    if count < 2:
        return glyph
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _area = (int(value) for value in stats[index])
    if h < 0.4 * height:
        return glyph
    return _Glyph(glyph.x0 + x, glyph.y0 + y, glyph.x0 + x + w, glyph.y0 + y + h, labels[y : y + h, x : x + w] == index)


def _circle_digit(glyph: _Glyph, ink: np.ndarray) -> _Glyph | None:
    """圈里的数字。数字常贴着圈，连通块分不开，按几何切：

    圈的直径取字框的高（后面粘着的延音弧、滑音线只会让框变宽），圆心在左边往右半个直径处；
    量出圈的线宽，只留内圆里的墨。GP 画圈时把圈里涂白，弦线不会穿过数字。
    """
    height = glyph.height
    radius = height / 2.0
    cy, cx = glyph.y0 + radius, glyph.x0 + radius
    column = int(round(cx))
    if column >= ink.shape[1]:
        return None
    # 圈顶的线宽：正上方那一列从上往下第一段墨的长度。
    top_run = 0
    for row in range(glyph.y0, glyph.y1):
        if ink[row, column]:
            top_run += 1
        elif top_run:
            break
    if not 1 <= top_run <= 0.25 * height:
        return None
    inner = radius - top_run - 0.3
    y0, y1 = glyph.y0, glyph.y1
    x0, x1 = glyph.x0, min(ink.shape[1], glyph.x0 + height)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    disk = (ys + 0.5 - cy) ** 2 + (xs + 0.5 - cx) ** 2 < inner**2
    interior = ink[y0:y1, x0:x1].astype(bool) & disk
    # 圈的两侧比顶上粗，内圆边上还会剩几小截圈。整块都贴着内圆边的是圈的残段，去掉。
    distance = np.sqrt((ys + 0.5 - cy) ** 2 + (xs + 0.5 - cx) ** 2)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(interior.astype(np.uint8), connectivity=8)
    body = np.zeros_like(interior)
    for index in range(1, count):
        part = labels == index
        if float(distance[part].min()) < inner - 2.5:
            body |= part
    # 连着数字的圈残段只有一个像素宽，数字笔画至少两个像素：2×2 开运算后剩下的是数字，再按它把笔画找回来。
    solid = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if solid.any():
        body &= cv2.dilate(solid, np.ones((3, 3), np.uint8)).astype(bool)
    rows = np.flatnonzero(body.any(axis=1))
    cols = np.flatnonzero(body.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    mask = body[top:bottom, left:right]
    return _Glyph(x0 + left, y0 + top, x0 + right, y0 + bottom, mask)


def _looks_like_rest(glyph: _Glyph) -> bool:
    """GP 画在 TAB 里的八分、十六分休止符：像 7，但左上角是一个实心圆头。"""
    height, width = glyph.mask.shape
    corner = glyph.mask[: max(1, height // 3), : max(1, width // 2)]
    return float(corner.mean()) >= 0.55


def _rejoin(
    pieces: list[_Glyph],
    ink: np.ndarray,
    reach: int,
    band: np.ndarray,
    max_width: float | None = None,
) -> list[_Glyph]:
    """被抹线切成上下两截的字拼回去：左右大半重叠、中间隔着的正好是抹掉的弦线那几行。

    隔着的不是弦线（比如底弦数字和它下面的符干碎块），不拼。比一个字还宽的块（圈）不拼。
    """
    line_rows = band[:, 0].astype(bool)
    pieces = sorted(pieces, key=lambda piece: piece.y0)
    merged = True
    while merged:
        merged = False
        for i, upper in enumerate(pieces):
            for j in range(i + 1, len(pieces)):
                lower = pieces[j]
                if max_width is not None and max(upper.width, lower.width) > max_width:
                    continue
                gap = lower.y0 - upper.y1
                # 外框可以有些重叠：9 的圈和尾巴、圈起来的数字被切开后，两块的外框上下交错。
                if gap < -0.8 * min(upper.height, lower.height) or gap > 2 * reach + 2:
                    continue
                # 中间隔着的行（重叠或紧挨着时看切口附近）得是弦线那几行；
                # 或者只隔一两行：细字体的笔画在抗锯齿处断开（5 的短竖）。
                rows = line_rows[min(lower.y0, upper.y1) - 1 : max(lower.y0, upper.y1) + 1]
                if not rows.any() and not 0 < gap <= max(2, reach):
                    continue
                overlap = min(upper.x1, lower.x1) - max(upper.x0, lower.x0)
                if overlap < 0.4 * min(upper.width, lower.width):
                    continue
                pieces[i] = _union(upper, lower, ink)
                pieces.pop(j)
                merged = True
                break
            if merged:
                break
    return pieces


def _union(upper: _Glyph, lower: _Glyph, ink: np.ndarray) -> _Glyph:
    x0, y0 = min(upper.x0, lower.x0), min(upper.y0, lower.y0)
    x1, y1 = max(upper.x1, lower.x1), max(upper.y1, lower.y1)
    mask = np.zeros((y1 - y0, x1 - x0), bool)
    mask[upper.y0 - y0 : upper.y1 - y0, upper.x0 - x0 : upper.x1 - x0] |= upper.mask
    mask[lower.y0 - y0 : lower.y1 - y0, lower.x0 - x0 : lower.x1 - x0] |= lower.mask
    # 抹掉的那几行里，上下都连着笔画的列补回来（竖笔、斜笔）。
    gap_top, gap_bottom = upper.y1 - y0, lower.y0 - y0
    if gap_bottom > gap_top:
        above = mask[gap_top - 1] if gap_top > 0 else np.zeros(mask.shape[1], bool)
        below = mask[gap_bottom] if gap_bottom < mask.shape[0] else np.zeros(mask.shape[1], bool)
        grown_above = _widen(above)
        grown_below = _widen(below)
        bridge = ink[upper.y1 : lower.y0, x0:x1].astype(bool) & (grown_above & grown_below)[None, :]
        mask[gap_top:gap_bottom] |= bridge
    return _Glyph(x0, y0, x1, y1, mask)


def _widen(row: np.ndarray) -> np.ndarray:
    """一行布尔值左右各扩一格。"""
    grown = row.copy()
    grown[1:] |= row[:-1]
    grown[:-1] |= row[1:]
    return grown


def _groups(row: list[_Glyph], typical: float) -> list[list[_Glyph]]:
    """同一根弦上左右紧挨着的几个字是一个品位（10、12、17），括号也并进来。"""
    groups: list[list[_Glyph]] = []
    for glyph in row:
        if groups:
            last = groups[-1][-1]
            if glyph.x0 - last.x1 <= 0.55 * typical:
                groups[-1].append(glyph)
                continue
        groups.append([glyph])
    return groups


def _read_group(
    group: list[_Glyph],
    region: np.ndarray,
    string: int,
    bank: Bank,
    typical: float | None = None,
) -> list[FretNote]:
    """一组挨着的字：逐个认，在滑音斜线处拆成前后两个音，去掉括号。

    认不出的字、字母（TAB 谱号）会让它所在的那一段作废，不影响斜线另一边。
    """
    read: list[tuple[_Glyph, str | None, float, np.ndarray]] = []
    # 搭着弧线的数字：弧线往右伸是 "start"（从它开始击弦、勾弦），往左伸是 "stop"。
    arcs: dict[int, str] = {}
    for glyph in group:
        # 一两像素宽的细条（斜线、弦线抹剩的边）不是字，也不能把 "11" 隔成两个 "1"。
        if typical and glyph.width < 0.15 * typical:
            continue
        # 左右大半落在同组另一个更宽的块里的小块，是那个字（X、0）被切下来的一笔，不是斜线，也不单独算字。
        if any(
            other is not glyph and other.width > glyph.width and min(glyph.x1, other.x1) - max(glyph.x0, other.x0) >= 0.6 * glyph.width
            for other in group
        ):
            continue
        # 夹在两个数字中间、比数字矮一大截的短斜线是滑音（GP 画得很短，常被认成 7）。
        slash = _slash_direction(glyph) if typical and glyph.height < 0.7 * typical and glyph.width < 1.6 * typical else None
        if slash is not None:
            read.append((glyph, slash, 0.0, np.zeros((20, 12), np.float32)))
            continue
        label, score, canvas = _read_glyph(glyph, region, bank)
        # 被弦线穿过的半截斜线也像 X：没把握、又看不出两道交叉斜笔的不算 X。
        if label in ("x", "X") and score < 0.7 and not _looks_like_x(glyph):
            label = None
        # 整首自适应的 7 模板只剩一道斜笔时，斜线也会被很有把握地认成 7：7 顶上有一道横，斜线没有。
        if label == "7" and typical and glyph.height < 1.15 * typical and not _top_bar(glyph):
            slash = _slash_direction(glyph) if _single_stroke(glyph) else None
            if slash is not None:
                read.append((glyph, slash, 0.0, np.zeros((20, 12), np.float32)))
                continue
        # 滑出的斜线（数字后面拖着的"\\"、"/"）和数字差不多高。认不成有把握的数字、又像斜线，就是它。
        # 闷音的 X 也是斜笔，认成 X 的不算。
        if typical and glyph.height < 1.15 * typical and glyph.width < 1.6 * typical and label not in ("x", "X") and (label is None or score < 0.7):
            slash = _slash_direction(glyph) if _single_stroke(glyph) else None
            if slash is not None:
                read.append((glyph, slash, 0.0, np.zeros((20, 12), np.float32)))
                continue
        if label is None and glyph.trimmed is not None and glyph.trimmed is not glyph:
            label, score, canvas = _read_glyph(glyph.trimmed, region, bank)
        if label is None and typical and glyph.height > 1.25 * typical:
            # 击勾弦的弧线从上面搭在数字上，连成一个高块：只留下面一个字高再认。
            cut = _below_arc(glyph, typical)
            if cut is not None:
                cut_label, cut_score, cut_canvas = _read_glyph(cut, region, bank)
                if cut_label is not None and cut_label.isdigit():
                    arcs[id(cut)] = _arc_side(glyph, cut)
                    glyph, label, score, canvas = cut, cut_label, cut_score, cut_canvas
        if label is None and typical and glyph.width > 0.9 * typical:
            # 斜线和数字粘成了一块（"\2"、"5/"）：竖着切开，一半是斜线、一半认得出数字就拆成两块。
            pair = _split_slash(glyph, region, bank)
            if pair is not None:
                read.extend(pair)
                continue
            # 斜线斜穿过后面那个数字（"(5)\2"）：切不出认得的数字，但紧跟在一个音后面、左半截是完整的一道斜线，
            # 斜线记到前一个音上；剩下的当认不出的块（五线谱上有音时按符头补）。
            previous = read[-1][1] if read else None
            prefix = _slash_prefix(glyph, typical) if previous is not None and (previous.isdigit() or previous == ")") else None
            if prefix is not None:
                read.append(prefix)
                read.append((glyph, None, score, canvas))
                continue
        if label is None and _looks_like_x(glyph):
            # 闷音的 X 被弦线穿过、笔画粗细不一，模板对不上，看形状：两道斜笔交叉。
            label, score = "X", 0.6
        if label is None:
            # 数字和旁边拖出去的滑音线、弦线残段拼成了一块：只取最大的一块连通墨再认。
            main = _largest_part(glyph)
            if main is not None:
                main_label, main_score, main_canvas = _read_glyph(main, region, bank)
                if main_label is not None and main_label.isdigit():
                    glyph, label, score, canvas = main, main_label, main_score, main_canvas
        if label is None and bank is not tab_bank():
            # 整首自适应的模板里有和它很像的类、分不开时，退回通用模板再认一次（只出现一两次的字形）。
            label, score, canvas = _read_glyph(glyph, region, tab_bank())
        if label == "7" and _looks_like_rest(glyph):
            label = "rest"
        read.append((glyph, label, score, canvas))
    # 淡的括号常被认成 1：比一般数字高出一截、又紧挨着数字的 "1"，在数字左边是 "("，右边是 ")"。
    if typical:
        for index, (glyph, label, score, canvas) in enumerate(read):
            if label != "1" or glyph.height < 1.15 * typical:
                continue
            after = read[index + 1][1] if index + 1 < len(read) else None
            before = read[index - 1][1] if index else None
            if after is not None and after.isdigit():
                read[index] = (glyph, "(", score, canvas)
            elif before is not None and before.isdigit():
                read[index] = (glyph, ")", score, canvas)
    segments: list[list[tuple[_Glyph, str | None, float, np.ndarray]]] = [[]]
    slides: list[tuple[str, float] | None] = [None]
    leading: str | None = None
    for item in read:
        # 滑音斜线、休止符、认不出的块都把一组拆开：两边各自成音，认不出的那块不牵连别人。
        if item[1] in ("/", "\\", "rest", None):
            if item[1] in ("/", "\\"):
                if any(segments):
                    # 挂到前面最近的一个音上（")" 之后已经开了一个空段）。
                    last = max(index for index, segment in enumerate(segments) if segment)
                    slides[last] = (item[1], (item[0].x0 + item[0].x1) / 2.0)
                else:
                    # 前面还没有音：滑入到后面那个音。
                    leading = item[1]
            segments.append([])
            slides.append(None)
            continue
        # 左括号开始一个新音，右括号结束一个音："(11)11" 是两个音。
        if item[1] == "(" and segments[-1]:
            segments.append([])
            slides.append(None)
        current = segments[-1]
        # 品位最多两位："(" 后面的第三个字只能是右括号（它常被认成 1）。
        if current and current[0][1] == "(" and len(current) == 3 and item[1] != ")":
            item = (item[0], ")", item[2], item[3])
        current.append(item)
        if item[1] == ")":
            # 以 ")" 收尾、前面没有 "("、开头却是个 "1"：淡的左括号只剩一截竖笔。
            if len(current) >= 3 and current[0][1] == "1" and not any(part[1] == "(" for part in current):
                current[0] = (current[0][0], "(", current[0][2], current[0][3])
            segments.append([])
            slides.append(None)
    split_segments: list[list[tuple[_Glyph, str | None, float, np.ndarray]]] = []
    split_slides: list[tuple[str, float] | None] = []
    for segment, slide in zip(segments, slides):
        pieces = _split_run(segment)
        split_segments.extend(pieces)
        split_slides.extend([None] * (len(pieces) - 1) + [slide])
    segments, slides = split_segments, split_slides
    notes: list[FretNote] = []
    for index, segment in enumerate(segments):
        note = _segment_note(segment, string)
        if note is None:
            continue
        # 斜线在这个音后面：后面还有音就是从它滑到下一个音，没有就是滑出。
        if slides[index] is not None:
            note.slide, note.slide_x = slides[index]
        note.arc = next((arcs[id(item[0])] for item in segment if id(item[0]) in arcs), None)
        digits = [item[0] for item in segment if item[1] is not None and item[1].isdigit()]
        # 括号里的延音不算（括号比数字高，把字高拉高了）；一两个像素宽的细竖条是小节线、弦线的碎段，也不算。
        # 小一号的字是整个按比例缩小的，高宽比不变（1.4 以上）；被弦线、光标削掉一截的普通数字只是变矮，宽度不变。
        note.small = bool(
            typical
            and digits
            and not note.paren
            and all(
                glyph.height < 0.8 * typical and glyph.width >= max(3, 0.2 * typical) and glyph.height >= 1.3 * glyph.width for glyph in digits
            )
        )
        if not notes:
            note.slide_in = leading
        notes.append(note)
    for note in notes[:-1]:
        note.slide_next = note.slide is not None
    return notes


def _below_arc(glyph: _Glyph, typical: float) -> _Glyph | None:
    """高块里下面一个字高的那一截（去掉搭在上面的弧线），左右收到有墨的列。"""
    keep = int(round(1.05 * typical))
    if glyph.height <= keep:
        return None
    part = glyph.mask[glyph.height - keep :]
    # 只留最大的一块（数字本身）：旁边断开的弦线残段、弧线的尾巴不要。
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(part.astype(np.uint8), connectivity=8)
    if count < 2:
        return None
    part = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    columns = np.flatnonzero(part.any(axis=0))
    rows = np.flatnonzero(part.any(axis=1))
    if columns.size == 0 or rows.size == 0:
        return None
    part = part[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    y0 = glyph.y1 - keep + int(rows[0])
    x0 = glyph.x0 + int(columns[0])
    return _Glyph(x0, y0, x0 + part.shape[1], y0 + part.shape[0], part, glyph.string)


def _looks_like_x(glyph: _Glyph) -> bool:
    """两道斜笔交叉的 X：宽高差不多，缩到 16×16 后两条对角线上几乎都是墨，上下边正中、左右边正中是空的。"""
    if not 0.6 <= glyph.width / max(1, glyph.height) <= 1.6 or glyph.height < 6:
        return False
    small = cv2.resize(glyph.mask.astype(np.uint8), (16, 16), interpolation=cv2.INTER_AREA) > 0
    grown = cv2.dilate(small.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    steps = range(2, 14)
    main = np.mean([grown[i, i] for i in steps])
    anti = np.mean([grown[i, 15 - i] for i in steps])
    edges = [small[0:3, 6:10], small[13:16, 6:10], small[6:10, 0:3], small[6:10, 13:16]]
    return bool(main >= 0.85 and anti >= 0.85 and all(edge.mean() < 0.25 for edge in edges))


def _crop(glyph: _Glyph, x0: int, x1: int) -> _Glyph | None:
    """块里 [x0, x1) 这几列，收紧到有墨的范围。"""
    part = glyph.mask[:, x0:x1]
    rows = np.flatnonzero(part.any(axis=1))
    columns = np.flatnonzero(part.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    part = part[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    left, top = glyph.x0 + x0 + int(columns[0]), glyph.y0 + int(rows[0])
    return _Glyph(left, top, left + part.shape[1], top + part.shape[0], part, glyph.string)


def _split_slash(glyph: _Glyph, region: np.ndarray, bank: Bank) -> list[tuple[_Glyph, str | None, float, np.ndarray]] | None:
    """一块切成左右两半：一半是一笔斜线、另一半认得出数字（相关值 0.7 以上）。斜线在左右都试。"""
    best = None
    for cut in range(int(0.3 * glyph.width), int(0.7 * glyph.width) + 1):
        left, right = _crop(glyph, 0, cut), _crop(glyph, cut, glyph.width)
        if left is None or right is None:
            continue
        for slash_part, digit_part, slash_first in ((left, right, True), (right, left, False)):
            direction = _slash_direction(slash_part) if _single_stroke(slash_part) else None
            if direction is None:
                continue
            label, score, canvas = _read_glyph(digit_part, region, bank)
            if label is None or not label.isdigit() or score < 0.7:
                continue
            if best is None or score > best[0]:
                slash_item = (slash_part, direction, 0.0, np.zeros((20, 12), np.float32))
                digit_item = (digit_part, label, score, canvas)
                best = (score, [slash_item, digit_item] if slash_first else [digit_item, slash_item])
    return best[1] if best else None


def _slash_prefix(glyph: _Glyph, typical: float) -> tuple[_Glyph, str, float, np.ndarray] | None:
    """块的左半截（切在三成到七成宽之间）是不是完整的一道斜线：一笔、够高（六成字高以上）、斜得出方向。"""
    for cut in range(int(0.7 * glyph.width), int(0.3 * glyph.width) - 1, -1):
        part = _crop(glyph, 0, cut)
        if part is None or part.height < 0.6 * typical or not _single_stroke(part):
            continue
        direction = _slash_direction(part)
        if direction is not None:
            return (part, direction, 0.0, np.zeros((20, 12), np.float32))
    return None


def _largest_part(glyph: _Glyph) -> _Glyph | None:
    """块里最大的一块连通墨（有不止一块时），收紧到它的范围。"""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(glyph.mask.astype(np.uint8), connectivity=8)
    if count < 3:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _area = (int(value) for value in stats[index])
    part = labels[y : y + h, x : x + w] == index
    return _Glyph(glyph.x0 + x, glyph.y0 + y, glyph.x0 + x + w, glyph.y0 + y + h, part, glyph.string)


def _arc_side(glyph: _Glyph, digit: _Glyph) -> str:
    """去掉的那截弧线在数字右边多还是左边多。"""
    top = glyph.mask[: digit.y0 - glyph.y0]
    ys, xs = np.nonzero(top)
    centre = (digit.x0 + digit.x1) / 2.0 - glyph.x0
    return "start" if xs.size and float(xs.mean()) > centre else "stop"


def _split_run(
    segment: list[tuple[_Glyph, str | None, float, np.ndarray]],
) -> list[list[tuple[_Glyph, str | None, float, np.ndarray]]]:
    """挨得很近的一串数字（"10-10-10"、"8-10-8"）被并成一段：品位最多两位，三个以上时从最宽的空隙开始拆，
    拆到每段不超过两位为止（两位数里两个字挨得最近）。空隙按字的中心量："1" 很窄，按边量 "12" 里面的空反而显得大。"""
    if len(segment) <= 2 or not all(label is not None and label.isdigit() for _g, label, _s, _c in segment):
        return [segment]
    if 4 <= len(segment) <= 12:
        even = _even_split(segment)
        if even is not None:
            return even
    gaps = [(b[0].x0 + b[0].x1) / 2.0 - (a[0].x0 + a[0].x1) / 2.0 for a, b in zip(segment, segment[1:])]
    cuts: set[int] = set()
    for index in sorted(range(len(gaps)), key=lambda i: -gaps[i]):
        cuts.add(index)
        pieces = [[segment[0]]]
        for position, item in enumerate(segment[1:]):
            if position in cuts:
                pieces.append([])
            pieces[-1].append(item)
        if all(len(piece) <= 2 for piece in pieces):
            return pieces
    return [segment]


def _even_split(segment: list) -> list[list] | None:
    """四个字以上的一串：在所有合法的拆法里（两位数不超过 24、不以 0 开头）挑各音间距最均匀、两位数里两个字又挨得够近的。
    排得很密的十六分音符（"12-12-12-12"）两个音之间的空比 "12" 里面还小，按空隙拆不对，但连着的音间距是均匀的。"""
    centres = [(item[0].x0 + item[0].x1) / 2.0 for item in segment]
    labels = [item[1] for item in segment]
    best: tuple[float, int, list[tuple[int, int]]] | None = None

    def walk(start: int, pieces: list[tuple[int, int]]) -> None:
        nonlocal best
        if start == len(segment):
            middles = [(centres[a] + centres[b - 1]) / 2.0 for a, b in pieces]
            steps = np.diff(middles)
            cost = float(np.std(steps) / max(1e-6, np.mean(steps)))
            # 两位数里两个字的距离要明显比音和音的间距小。
            inner = [centres[b - 1] - centres[a] for a, b in pieces if b - a == 2]
            if inner:
                cost += 2.0 * max(0.0, max(inner) / max(1e-6, float(steps.min())) - 0.65)
            if best is None or (cost, len(pieces)) < (best[0], best[1]):
                best = (cost, len(pieces), list(pieces))
            return
        walk(start + 1, pieces + [(start, start + 1)])
        if start + 2 <= len(segment) and labels[start] != "0" and int(labels[start] + labels[start + 1]) <= 24:
            walk(start + 2, pieces + [(start, start + 2)])

    walk(0, [])
    if best is None:
        return None
    return [segment[a:b] for a, b in best[2]]


def _single_stroke(glyph: _Glyph) -> bool:
    """只有一道笔画：几乎每一行只有一段墨，墨也很稀。淡的 9、4 带圈带横，过不了这一关。"""
    mask = glyph.mask
    if mask.size == 0 or float(mask.mean()) > 0.35:
        return False
    runs = [int(np.count_nonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))) == 1)) for row in mask]
    busy = sum(1 for count in runs if count > 1)
    return busy <= 0.15 * len(runs)


def _slash_direction(glyph: _Glyph) -> str | None:
    """斜线：上半截和下半截的墨左右错开。上半截偏右是 "/"（往上滑），偏左是 "\\"。不像斜线返回 None。"""
    # 陡的斜线也就高的一半宽；1 是竖直的，下面上下两半错不开，会被排除。
    if glyph.width < 0.4 * glyph.height:
        return None
    # 斜线常被一小段弦线从中间横穿，中间几乎整行都是墨的那一行不算。7 的上横在最顶上，留着。
    full = glyph.mask.mean(axis=1) >= 0.8
    rows = np.arange(glyph.height)
    full &= (rows >= 0.25 * glyph.height) & (rows < 0.75 * glyph.height)
    mask = glyph.mask & ~full[:, None]
    ys, xs = np.nonzero(mask)
    middle = glyph.height / 2.0
    upper, lower = xs[ys < middle], xs[ys >= middle]
    if upper.size == 0 or lower.size == 0:
        return None
    shift = float(upper.mean() - lower.mean())
    if abs(shift) < max(1.0, 0.12 * glyph.width):
        return None
    return "/" if shift > 0 else "\\"


def _read_glyph(glyph: _Glyph, region: np.ndarray, bank: Bank) -> tuple[str | None, float, np.ndarray]:
    patch = np.where(glyph.mask, region[glyph.y0 : glyph.y1, glyph.x0 : glyph.x1], 0)
    canvas = fit_gray(patch)
    label, score, margin = classify(canvas, bank, narrow=glyph.width < 0.5 * glyph.height)
    if label is None or score < 0.8 or margin < MIN_MARGIN:
        # 字边上粘着一两个像素的残渣（弦线、背景抹剩的）：字框宽出一列，缩放后笔画整体错开，相关值掉得厉害。
        # 去掉边上只有一两个像素的列再认一次，更像才用。
        trimmed = _trim_sparse_edges(glyph)
        if trimmed is not None:
            patch2 = np.where(trimmed.mask, region[trimmed.y0 : trimmed.y1, trimmed.x0 : trimmed.x1], 0)
            canvas2 = fit_gray(patch2)
            label2, score2, margin2 = classify(canvas2, bank, narrow=trimmed.width < 0.5 * trimmed.height)
            if label2 is not None and score2 >= MIN_SCORE and margin2 >= MIN_MARGIN and score2 > score:
                return label2, score2, canvas2
    if label is None or score < MIN_SCORE or margin < MIN_MARGIN:
        return None, score, canvas
    return label, score, canvas


def _trim_sparse_edges(glyph: _Glyph) -> _Glyph | None:
    """去掉左右边上墨不到两个像素的列；没有可去的返回 None。"""
    counts = glyph.mask.sum(axis=0)
    left, right = 0, glyph.mask.shape[1]
    limit = max(1, int(0.08 * glyph.height))
    while right - left > 2 and counts[left] <= limit:
        left += 1
    while right - left > 2 and counts[right - 1] <= limit:
        right -= 1
    if left == 0 and right == glyph.mask.shape[1]:
        return None
    return _crop(glyph, left, right)


def _segment_note(
    segment: list[tuple[_Glyph, str | None, float, np.ndarray]],
    string: int,
) -> FretNote | None:
    paren = False
    while segment and segment[0][1] == "(":
        segment = segment[1:]
        paren = True
    while segment and segment[-1][1] == ")":
        segment = segment[:-1]
        paren = True
    if len(segment) == 1 and segment[0][1] in ("x", "X"):
        glyph, _label, score, canvas = segment[0]
        return FretNote(string + 1, 0, (glyph.x0 + glyph.x1) / 2.0, glyph.x0, glyph.x1, score, paren, dead=True, canvases=[canvas])
    if not 1 <= len(segment) <= 2 or not all(label is not None and label.isdigit() for _g, label, _s, _c in segment):
        return None
    fret = int("".join(str(label) for _g, label, _s, _c in segment))
    if fret > 30:
        return None
    left, right = segment[0][0].x0, segment[-1][0].x1
    return FretNote(
        string=string + 1,
        fret=fret,
        x=(left + right) / 2.0,
        left=left,
        right=right,
        score=min(score for _g, _l, score, _c in segment),
        paren=paren,
        canvases=[canvas for _g, _l, _s, canvas in segment],
    )


def classify(canvas: np.ndarray, bank: Bank, narrow: bool | None = None) -> tuple[str | None, float, float]:
    """(字, 最像的相关值, 和第二像的数字的差距)。

    数字之间比差距。字母、括号要比最像的数字明显更像才算：1 和 T、) 本来就长得近。
    narrow 给出字形宽高：窄（宽不到高的一半）的数字只能是 1，宽的不会是 1。
    很多字体里 1 和 7 的相关值只差一点点，靠宽窄一下就分开了。
    """
    scores: dict[str, float] = {}
    for (label, _template), score in zip(bank, corr_all(canvas, bank)):
        if score > scores.get(label, -1.0):
            scores[label] = score
    if narrow is not None:
        for label in list(scores):
            if label.isdigit() and (label == "1") != narrow:
                del scores[label]
    digits = sorted(((label, score) for label, score in scores.items() if label.isdigit()), key=lambda item: -item[1])
    others = sorted(((label, score) for label, score in scores.items() if not label.isdigit()), key=lambda item: -item[1])
    if not digits:
        return None, 0.0, 0.0
    if others and others[0][1] >= digits[0][1] + 0.05:
        return others[0][0], others[0][1], others[0][1] - digits[0][1]
    second = digits[1][1] if len(digits) > 1 else -1.0
    if digits[0][1] - second >= _CLEAR_MARGIN or len(digits) < 2:
        return digits[0][0], digits[0][1], digits[0][1] - second
    return _chamfer_decide(canvas, bank, [label for label, _score in digits[:3]], dict(digits))


# 相关值差距到这么大就直接用；不到就改用倒角距离在前几名里裁决。
_CLEAR_MARGIN = 0.08


def _chamfer_decide(
    canvas: np.ndarray,
    bank: Bank,
    candidates: list[str],
    scores: dict[str, float],
) -> tuple[str | None, float, float]:
    """前几名相关值很接近时，比笔画的倒角距离：两边的笔画各自离对方最近的笔画有多远，取平均。

    相关值对"多出一段笔画"不敏感（平顶的 3 和 8 只差 0.04）；倒角距离对它很敏感。
    返回的差距换算成和相关值差距可比的量，交给上面的门槛判断。
    """
    ink = canvas >= 0.4
    if not ink.any():
        return None, 0.0, 0.0
    distance_to_ink = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    best: dict[str, float] = {}
    for label, template in bank:
        if label not in candidates:
            continue
        stroke = template >= 0.4
        if not stroke.any():
            continue
        distance_to_stroke = cv2.distanceTransform((~stroke).astype(np.uint8), cv2.DIST_L2, 3)
        value = 0.5 * (float(distance_to_stroke[ink].mean()) + float(distance_to_ink[stroke].mean()))
        best[label] = min(best.get(label, float("inf")), value)
    ranked = sorted(best.items(), key=lambda item: item[1])
    if len(ranked) < 2:
        return None, 0.0, 0.0
    (label, value), (_second, other) = ranked[0], ranked[1]
    # 按相对差距算：领先 20% 换算成 0.04，正好是门槛。和相关值的第一名不一致时要领先更多。
    relative = (other - value) / max(other, 1e-6)
    if label != candidates[0]:
        relative -= 0.15
    return label, scores.get(label, 0.0), relative * 0.2


def adapted_tab_bank(
    measures: list[tuple[np.ndarray, TabStaff, tuple[int, int] | None]],
    bank: Bank | None = None,
) -> Bank:
    """整首曲子一起看：把所有候选字形按形状聚类，给每一类定一个数字，类的平均字形做成这首曲子的专属模板。

    同一首曲子的品位数字字体、字号都一样，同一个数字的字形几乎一模一样。
    通用模板对某个字体分不清（MuseScore 的 9 和 0 只差 0.01），但按"排他"原则就能分开：
    把握大的类先占住自己的数字；把握小的类取第一个没被别的类（形状不同的类）占住的数字。
    """
    bank = bank or tab_bank()
    canvases: list[np.ndarray] = []
    for image, staff, bars in measures:
        spacing = staff.spacing
        y0 = max(0, int(staff.top - 0.8 * spacing))
        y1 = min(image.shape[0], int(staff.bottom + 0.8 * spacing) + 1)
        region = (255 - image[y0:y1]).astype(np.uint8)
        for glyph in _glyphs(region, [line - y0 for line in staff.lines], spacing):
            if bars is not None and any(glyph.x0 - 0.35 * spacing <= bar <= glyph.x1 + 0.35 * spacing for bar in bars):
                continue
            # 窄的是 1、括号，不参与：它们本来就好认。
            if glyph.width < 0.5 * glyph.height:
                continue
            patch = np.where(glyph.mask, region[glyph.y0 : glyph.y1, glyph.x0 : glyph.x1], 0)
            canvases.append(fit_gray(patch))
    clusters = _cluster(canvases)
    labelled: list[tuple[np.ndarray, str, float, float]] = []
    for members in clusters:
        if len(members) < 3:
            continue
        centroid = np.mean(np.stack([canvases[index] for index in members]), axis=0).astype(np.float32)
        scores = _digit_scores(centroid, bank)
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
        labelled.append((centroid, ranked[0][0], ranked[0][1], margin))
    # 先定把握大的，再定把握小的。
    claimed: list[tuple[str, np.ndarray]] = []
    own: list[tuple[str, np.ndarray]] = []
    for centroid, _label, top, margin in sorted(labelled, key=lambda item: -item[3]):
        scores = _digit_scores(centroid, bank)
        for label, score in sorted(scores.items(), key=lambda item: -item[1])[:3]:
            # 退到下一个数字只在它和最像的那个差不多、这一类字形本身又清楚时才行（MuseScore 的 9 和 0 只差 0.01）。不硬给：
            # 压在视频画面上糊了一点的 "8" 自成一小类，8、6 都被别的类占了，会一路退成 0。
            if score < MIN_SCORE or top - score >= _CLEAR_MARGIN:
                break
            if score < top and top < 0.8:
                # 本身就认不太清的一类（背景糊过的几个字形）不往下退。
                break
            taken = any(
                other_label == label and _glyph_corr(centroid, other) < 0.85 for other_label, other in claimed
            )
            if taken and margin < _CLEAR_MARGIN:
                continue
            claimed.append((label, centroid))
            own.append((label, centroid))
            break
    return tuple(own) + bank


def _digit_scores(canvas: np.ndarray, bank: Bank) -> dict[str, float]:
    scores: dict[str, float] = {}
    for (label, _template), score in zip(bank, corr_all(canvas, bank)):
        if label.isdigit() and score > scores.get(label, -1.0):
            scores[label] = score
    return scores


def _cluster(canvases: list[np.ndarray], threshold: float = 0.9) -> list[list[int]]:
    """按形状贪心聚类：和已有某一类的平均字形相关值够高就并进去。"""
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    for index, canvas in enumerate(canvases):
        best, best_score = -1, threshold
        for position, centroid in enumerate(centroids):
            score = _glyph_corr(canvas, centroid)
            if score >= best_score:
                best, best_score = position, score
        if best < 0:
            centroids.append(canvas.copy())
            members.append([index])
            continue
        members[best].append(index)
        count = len(members[best])
        centroids[best] = centroids[best] + (canvas - centroids[best]) / count
    return members


def onsets(notes: list[FretNote], spacing: float) -> list[Onset]:
    """横坐标差不到半个线距、又不在同一根弦上的，是同时发声的和弦。"""
    groups: list[Onset] = []
    for note in sorted(notes, key=lambda item: item.x):
        if groups:
            last = groups[-1]
            if abs(note.x - last.x) <= 0.5 * spacing and all(item.string != note.string for item in last.notes):
                last.notes.append(note)
                last.x = float(np.mean([item.x for item in last.notes]))
                continue
        groups.append(Onset(x=note.x, notes=[note]))
    for group in groups:
        group.notes.sort(key=lambda item: item.string)
    return groups


def vote_frets(readings: list[list[FretNote]], spacing: float) -> list[FretNote]:
    """同一小节几帧各读一遍，按位置投票。

    同一根弦上横坐标差不到 0.6 个线距的读数算同一个音；一半以上的帧读到它才留下（单帧的缺笔、
    背景里的碎块只影响一两帧）。品位按置信度加权取票数最多的，括号、滑音按多数。
    """
    if len(readings) <= 1:
        return list(readings[0]) if readings else []
    pooled = sorted(
        ((frame, note) for frame, notes in enumerate(readings) for note in notes),
        key=lambda item: (item[1].string, item[1].x),
    )
    clusters: list[list[tuple[int, FretNote]]] = []
    for frame, note in pooled:
        if clusters:
            last = clusters[-1]
            centre = float(np.median([item.x for _f, item in last]))
            if last[0][1].string == note.string and abs(note.x - centre) <= 0.6 * spacing:
                last.append((frame, note))
                continue
        clusters.append([(frame, note)])
    voted: list[FretNote] = []
    for cluster in clusters:
        frames = {frame for frame, _note in cluster}
        # 一半以上的帧读到才算。只有两帧、只有一帧读到时，要读得很有把握（背景里的细竖线常在一帧里被认成 1）。
        if len(frames) * 2 < len(readings):
            continue
        if len(readings) == 2 and len(frames) == 1 and max(note.score for _f, note in cluster) < 0.8:
            continue
        weights: dict[int, float] = {}
        for _frame, note in cluster:
            weights[note.fret] = weights.get(note.fret, 0.0) + max(note.score, 0.3)
        fret = max(weights, key=lambda value: weights[value])
        members = [note for _frame, note in cluster if note.fret == fret]
        best = max(members, key=lambda note: note.score)
        paren = sum(note.paren for _f, note in cluster) * 2 > len(cluster)
        slides = [note.slide for _f, note in cluster if note.slide]
        slide = max(set(slides), key=slides.count) if len(slides) * 2 >= len(cluster) else None
        slide_next = slide is not None and sum(note.slide_next for _f, note in cluster) * 2 >= len(cluster)
        ins = [note.slide_in for _f, note in cluster if note.slide_in]
        slide_in = max(set(ins), key=ins.count) if len(ins) * 2 >= len(cluster) else None
        small = sum(note.small for _f, note in cluster) * 2 > len(cluster)
        arcs = [note.arc for _f, note in cluster if note.arc]
        arc = max(set(arcs), key=arcs.count) if len(arcs) * 2 >= len(cluster) else None
        dead = sum(note.dead for _f, note in cluster) * 2 > len(cluster)
        voted.append(
            FretNote(
                string=best.string,
                fret=fret,
                x=float(np.median([note.x for note in members])),
                left=best.left,
                right=best.right,
                score=best.score,
                paren=paren,
                slide=slide,
                slide_next=slide_next,
                slide_in=slide_in,
                slide_x=float(np.median([note.slide_x for _f, note in cluster if note.slide_x is not None])) if slide and any(note.slide_x is not None for _f, note in cluster) else None,
                dead=dead,
                arc=arc,
                small=small,
                legato=sum(note.legato for _f, note in cluster) * 2 > len(cluster),
                slur_next=sum(note.slur_next for _f, note in cluster) * 2 > len(cluster),
                canvases=best.canvases,
                weak=len(frames) * 2 == len(readings),
                slide_weak=slide is not None and len(slides) * 2 == len(cluster),
            )
        )
    # 斜线的位置正落在同一根弦上另一个音的数字上：那是 7 的斜笔被切下来当成了斜线，不是滑音。
    for note in voted:
        if note.slide and note.slide_x is not None and any(
            other is not note and other.string == note.string and abs(other.x - note.slide_x) <= 0.35 * spacing for other in voted
        ):
            note.slide, note.slide_next, note.slide_x = None, False, None
    return sorted(voted, key=lambda note: (note.x, note.string))


def mark_arcs(image: np.ndarray, staff: TabStaff, notes: list[FretNote], bars: tuple[int, int] | None = None) -> None:
    """数字上方或下方分开画的细弧线（击弦、勾弦）：弧线底下盖住的同一根弦上的几个音，每个都连到下一个。

    在离弦线 0.55~1.5 个线距的那一条带里（先抹掉弦线）找细墨块：一个线距以上宽、不到一个线距高、墨很稀，
    而且是弯的（上方的弧中间高、两头低，下方的反过来）。一道弧线常跨三个音（装饰音、击弦、勾弦连着）。
    两音之间已经有滑音的不算（滑音也常带弧线）。
    """
    spacing = staff.spacing
    ink = ((255 - image) >= 110).astype(np.uint8)
    length = max(9, int(round(1.2 * spacing)))
    long_runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    band = np.zeros_like(ink)
    reach = max(2, int(round(0.1 * spacing)))
    for line in staff.lines:
        row = int(round(line))
        band[max(0, row - reach) : row + reach + 1] = 1
    cleaned = cv2.subtract(ink, long_runs & band)
    # 弧线最低处常压在下一根弦线上，抹线后断成两半：上下都有墨、中间只隔着弦线那几行的地方竖着接回去。
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * reach + 3)))
    left, right = bars if bars is not None else (0, ink.shape[1])
    for string in range(1, staff.strings + 1):
        row = sorted((note for note in notes if note.string == string), key=lambda note: note.x)
        if len(row) < 2:
            continue
        y = staff.lines[string - 1]
        for side in (-1, 1):
            y0 = max(0, int(round(y - 1.5 * spacing if side < 0 else y + 0.55 * spacing)))
            y1 = min(cleaned.shape[0], int(round(y - 0.55 * spacing if side < 0 else y + 1.5 * spacing)))
            if y1 - y0 < 3:
                continue
            window = cleaned[y0:y1, left:right].astype(bool)
            for x0, x1 in _arcs(window, side, spacing):
                covered = [note for note in row if left + x0 - 0.6 * spacing <= note.x <= left + x1 + 0.6 * spacing]
                for first, _second in zip(covered, covered[1:]):
                    if first.slide_next:
                        first.slur_next = True
                    else:
                        first.arc_next = True
            # 弦线下面的弧线常被截掉中段，只剩两头各一截往中间斜下去的细笔。
            if side > 0:
                for first, second in zip(row, row[1:]):
                    if not first.slur_next and not first.arc_next and _cut_arc(window, first.x - left, second.x - left, spacing):
                        if first.slide_next:
                            first.slur_next = True
                        else:
                            first.arc_next = True
    # 斜线上方 TAB 外面印着 "sl."：连奏滑音（GP 只写滑音；数字下面另画了弧线的才另写连线）。
    # 小节最后一个音的斜线（连到下一小节的音）也看。
    for note in notes:
        if note.slide and note.slide_x is not None and _legato_text(ink, staff, note.slide_x):
            note.legato = True


def _cut_arc(window: np.ndarray, xa: float, xb: float, spacing: float) -> bool:
    """被截掉中段的弧线（截图时那一块的有效范围到此为止，再往下是白的）：两个音之间只剩两截细笔，
    一截挨着前一个音、一截挨着后一个音，都往中间斜下去，底端齐平地断在同一行。"""
    if window.shape[0] < 3 or xb - xa < 0.8 * spacing:
        return False
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(window.astype(np.uint8), connectivity=8)
    lefts: list[int] = []
    rights: list[int] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if w < 0.3 * spacing or h < 0.3 * spacing or area > 0.3 * spacing * max(w, h):
            continue
        mask = labels[y : y + h, x : x + w] == index
        top = np.flatnonzero(mask[: max(1, h // 3)].any(axis=0)).mean()
        low = np.flatnonzero(mask[-max(1, h // 3) :].any(axis=0)).mean()
        centre = x + w / 2.0
        if low > top and xa - 0.3 * spacing <= centre <= xa + 0.9 * spacing:
            lefts.append(y + h)
        elif low < top and xb - 0.9 * spacing <= centre <= xb + 0.3 * spacing:
            rights.append(y + h)
    return any(abs(a - b) <= 1 for a in lefts for b in rights)


def _legato_text(ink: np.ndarray, staff: TabStaff, x: float) -> bool:
    """TAB 上方、斜线附近的 "sl."：一个细高的 l，左边挨着一个矮的 s 或右边挨着一个小点。

    H、P（击勾弦）是一整块、比 l 宽；五线谱伸下来的符干比一个线距长，都不算。
    """
    spacing = staff.spacing
    y0, y1 = max(0, int(staff.top - 2.5 * spacing)), max(0, int(staff.top - 0.3 * spacing))
    x0, x1 = max(0, int(x - 1.5 * spacing)), min(ink.shape[1], int(x + 1.5 * spacing))
    if y1 <= y0 or x1 <= x0:
        return False
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1], connectivity=8)
    pieces = [tuple(int(value) for value in stats[index]) for index in range(1, count) if stats[index][4] >= 2]
    for lx, ly, lw, lh, _area in pieces:
        if not (0.4 * spacing <= lh <= 1.0 * spacing and lw <= 0.3 * spacing):
            continue
        bottom = ly + lh
        for px, py, pw, ph, _ in pieces:
            if px == lx and py == ly:
                continue
            near_bottom = abs(py + ph - bottom) <= 0.2 * spacing
            s_left = near_bottom and 0 < lx - (px + pw) <= 0.4 * spacing and 0.2 * spacing <= ph <= 0.6 * spacing
            dot_right = near_bottom and 0 <= px - (lx + lw) <= 0.4 * spacing and max(pw, ph) <= 0.25 * spacing
            if s_left or dot_right:
                return True
    return False


def _arcs(window: np.ndarray, side: int, spacing: float, min_width: float = 1.0) -> list[tuple[int, int]]:
    """带里弯的细墨块的左右范围。min_width：至少多宽（线距）。"""
    out: list[tuple[int, int]] = []
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(window.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if w < min_width * spacing or h > 1.0 * spacing or area > 0.25 * spacing * w:
            continue
        mask = labels[:, x : x + w] == index
        # GP 的弧线是中间粗、两头尖的月牙：按外侧边缘量弯度（上方的弧看最上面一行，下方的弧看最下面一行）。
        pick = (lambda rows: rows.min()) if side < 0 else (lambda rows: rows.max())
        centres = np.array([pick(np.flatnonzero(column)) for column in mask.T if column.any()], float)
        if centres.size < 6:
            continue
        quarter = centres.size // 4
        ends = (centres[:quarter].mean() + centres[-quarter:].mean()) / 2.0
        middle = centres[quarter:-quarter].mean()
        # 上方的弧（side = -1）中间更靠上（行号小），下方的反过来；至少弯 0.08 个线距。
        if (ends - middle) * -side >= 0.08 * spacing:
            out.append((x, x + w))
    return out


def fill_frets(voted: list[FretNote], extra: list[FretNote], spacing: float, min_score: float = 0.7) -> list[FretNote]:
    """清洗过的小节图上读到、各帧投票里没有的音补进来（原帧里被背景花纹、光标盖住的数字）。

    只补不改：同一根弦 0.6 个线距以内已经有音的不动；清洗图读得不够有把握的不补。
    """
    # 清洗图上字形大小不准（线重画过），补进来的音不算装饰音。
    # 它那道斜线的位置上各帧其实读到了一个音：斜线是被读歪的数字，去掉。
    def cleaned(note: FretNote) -> FretNote:
        fake = note.slide_x is not None and any(other.string == note.string and abs(other.x - note.slide_x) <= 0.6 * spacing for other in voted)
        if fake:
            return replace(note, small=False, slide=None, slide_next=False, slide_x=None)
        return replace(note, small=False)

    # 原帧被插画盖住时，背景纹理常被读成一串没把握的 "1"（相关值 0.6 出头），票还能过半；或者两位数被盖掉一位，
    # 只读出 "10" 里的 "1"。清洗图在同一处读得很有把握（0.85 以上）就信清洗图：投票结果里没把握的、或者品位只是
    # 清洗图那个数的一部分的，落在清洗图那个数字范围里就换掉。
    sure = [note for note in extra if note.score >= 0.85]
    replaced = [
        note
        for note in voted
        if any(
            other.string == note.string
            and other.left - 0.6 * spacing <= note.x <= other.right + 0.6 * spacing
            # 清洗图把两位数拆成两个一位数时（字距被重画的线撑开）不能拿它换：位数不能比投票结果少。
            and len(str(other.fret)) >= len(str(note.fret))
            and (note.score < 0.75 or (note.fret != other.fret and str(note.fret) in str(other.fret)))
            for other in sure
        )
    ]
    voted = [note for note in voted if note not in replaced]
    for note in voted:
        if note.slide_weak and any(
            other.string == note.string and other.fret == note.fret and abs(other.x - note.x) <= 0.6 * spacing and other.slide is None
            for other in extra
        ):
            note.slide, note.slide_next, note.slide_x = None, False, None
    # 只有一半的帧读到的音（只有两帧时就是一帧），清洗图同一处读不出同一个品（或读得没把握）的，多半是背景纹理。
    voted = [
        note
        for note in voted
        if not note.weak
        or any(other.string == note.string and other.fret == note.fret and abs(other.x - note.x) <= 0.6 * spacing and other.score >= 0.85 for other in extra)
    ]
    added = [
        cleaned(note)
        for note in extra
        if note.score >= min_score and not any(other.string == note.string and abs(other.x - note.x) <= 0.6 * spacing for other in voted)
    ]
    # 反过来也一样：各帧把后面那个数字（7 的斜笔）读成了斜线，清洗图在斜线的位置上读出了一个音，这道斜线去掉。
    for note in voted:
        if note.slide and note.slide_x is not None and any(
            other.string == note.string and abs(other.x - note.slide_x) <= 0.35 * spacing for other in added
        ):
            note.slide, note.slide_next, note.slide_x = None, False, None
    return sorted(voted + added, key=lambda note: (note.x, note.string))


def bass_attacks(image: np.ndarray, staff: TabStaff, bars: tuple[int, int] | None = None) -> list[tuple[float, str]]:
    """TAB 上方的拍弦 "S"、勾弦 "P"（和击弦的 "H"）字母：(中心横坐标, "slap" / "pop" / "H")。

    字母正对着它那个音，一般在 TAB 第一线上方一个线距；同一处还有 "sl."、"H" 时往上叠。按字母大小（约 0.65 个线距高）
    挑出连通块，再按竖笔分：一根贯穿的竖笔、上半截有圈是 P；两根竖笔是 H；没有竖笔、上中下都有横的弯是 S。
    """
    spacing = staff.spacing
    y0 = max(0, int(staff.top - 3.5 * spacing))
    y1 = max(y0, int(staff.top - 0.2 * spacing))
    x0, x1 = (bars[0], bars[-1]) if bars else (0, image.shape[1])
    ink = (image[y0:y1, x0:x1] < 128).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    found: list[tuple[float, str]] = []
    for index in range(1, count):
        x, y, w, h = (int(value) for value in stats[index, :4])
        # 碰到带子上沿的是从上面伸下来的（五线谱离 TAB 近时的符干、符尾、符杠），不是浮在 TAB 上方的字。
        if y == 0:
            continue
        if not (0.5 * spacing <= h <= 0.9 * spacing and 0.3 * spacing <= w <= 0.7 * spacing):
            continue
        letter = _attack_letter(labels[y : y + h, x : x + w] == index)
        if letter is not None:
            found.append((x0 + x + w / 2.0, letter))
    return found


def _attack_letter(mask: np.ndarray) -> str | None:
    height, width = mask.shape
    # 竖笔是一整段连着的（S 右边那一列上下两截加起来也有那么长，但中间断开）。
    columns = np.array([_longest_run(mask[:, column]) >= 0.75 * height for column in range(width)])
    stems = 0
    previous = False
    for value in columns:
        if value and not previous:
            stems += 1
        previous = bool(value)
    if stems >= 2:
        return "H"
    if stems == 1:
        # P：竖笔右边只有上半截有墨（圈），下半截空着。
        stem = int(np.flatnonzero(columns)[-1])
        right = mask[:, stem + 1 :]
        if right.size == 0:
            return None
        # 最下面两行是衬线，不算。
        upper, lower = right[: height // 2].sum(), right[int(0.65 * height) : max(int(0.65 * height) + 1, height - 2)].sum()
        return "pop" if upper > 0 and lower <= 0.15 * upper else None
    # S：上、中、下三段都有墨，左上和右下各有一笔（反过来的 Z 字形走向）。
    thirds = [mask[int(i * height / 3) : int((i + 1) * height / 3)].any() for i in range(3)]
    if not all(thirds):
        return None
    left_top = mask[: height // 3, : width // 2].sum()
    right_bottom = mask[2 * height // 3 :, width // 2 :].sum()
    middle = mask[height // 3 : 2 * height // 3]
    # 中段从左上斜到右下：中段的墨大致在中间。
    return "slap" if left_top > 0 and right_bottom > 0 and middle.sum() > 0 else None


def vibratos(image: np.ndarray, staff: TabStaff, bars: tuple[int, int] | None = None) -> list[tuple[float, float]]:
    """TAB 上方的揉弦波浪线（GP 画在揉弦那个音的正上方，往右延伸）：(左端, 右端) 横坐标。

    扁（不到 0.6 个线距高）、长（1.5 个线距以上）的一块，上沿一起一伏、至少三个峰。
    """
    spacing = staff.spacing
    y0 = max(0, int(staff.top - 3.5 * spacing))
    y1 = max(y0, int(staff.top - 0.2 * spacing))
    x0, x1 = (bars[0], bars[-1]) if bars else (0, image.shape[1])
    ink = (image[y0:y1, x0:x1] < 128).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    found: list[tuple[float, float]] = []
    for index in range(1, count):
        x, y, w, h = (int(value) for value in stats[index, :4])
        if y == 0 or h > 0.6 * spacing or h < 0.2 * spacing or w < 1.5 * spacing:
            continue
        mask = labels[y : y + h, x : x + w] == index
        tops = np.where(mask.any(axis=0), mask.argmax(axis=0), h)
        # 峰和峰之间要真的落下去（落差四成高以上）：微微倾斜的符杠上沿也会一段段碰到最上面一行。
        peaks = 0
        dropped = True
        for value in tops:
            if value <= 1 and dropped:
                peaks += 1
                dropped = False
            elif value >= 0.4 * h:
                dropped = True
        if peaks >= 3:
            found.append((float(x0 + x), float(x0 + x + w)))
    return found


def bends(image: np.ndarray, staff: TabStaff, bank: Bank, bars: tuple[int, int] | None = None) -> list[tuple[float, float]]:
    """TAB 上方的推弦箭头：(箭头尖的横坐标, 推多少半音)。

    GP 从音的右边画一道往上弯的线，顶上一个实心箭头，箭头上面标推几个全音（¼、½、¾、full）。弯线常跨过小节线，
    箭头落在下一小节里，所以这里只找箭头和标注，归到哪个音由 tabscore 按位置往前找。
    箭头：一块往上越来越窄的实心三角（顶上一两像素宽，往下半个线距内宽到 0.25 个线距以上）。
    标注：箭头上方 1.5 个线距以内的两个小数字（分子、分母）；没有数字的是 full。MusicXML 的 bend-alter 是半音数，等于全音数的两倍。
    """
    spacing = staff.spacing
    y0 = max(0, int(staff.top - 4.0 * spacing))
    y1 = max(y0, int(staff.top - 0.2 * spacing))
    x0, x1 = (bars[0], bars[-1]) if bars else (0, image.shape[1])
    ink = (image[y0:y1, x0:x1] < 128).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = [tuple(int(value) for value in stats[index, :4]) for index in range(1, count)]
    found: list[tuple[float, float]] = []
    for index in range(1, count):
        x, y, w, h = boxes[index - 1]
        if y == 0 or h < 1.0 * spacing:
            continue
        mask = labels[y : y + h, x : x + w] == index
        head = mask[: max(3, int(round(0.5 * spacing)))]
        widths = head.sum(axis=1)
        if widths[0] > 3 or widths.max() < 0.25 * spacing:
            continue
        # 三角是实心的：尖上细的只有一两行就开始变宽（往上的符干细得长），最宽那一行以上每行都连成一段。
        thin = int(np.argmax(widths > 2)) if (widths > 2).any() else len(widths)
        widest = int(np.argmax(widths))
        if thin > 0.2 * spacing + 1 or any(_runs(row) > 1 for row in head[: widest]):
            continue
        tip_x = x0 + x + float(np.flatnonzero(mask[0]).mean())
        tip_y = y
        above = [
            box
            for box in boxes
            if 0.25 * spacing <= box[3] <= 0.8 * spacing
            and tip_y - 1.5 * spacing <= box[1] + box[3] <= tip_y
            and abs(x0 + box[0] + box[2] / 2.0 - tip_x) <= 1.0 * spacing
        ]
        # 箭头上面总印着幅度（分数或 full）：一个字都没有的不是推弦。
        if not above:
            continue
        digits = [box for box in above if box[3] <= 0.55 * spacing]
        amount = 2.0
        if len(digits) >= 2:
            # 分子在左上、分母在右下。
            digits.sort(key=lambda box: box[0])
            read = []
            for bx, by, bw, bh in (digits[0], digits[-1]):
                glyph = ink[by : by + bh, bx : bx + bw] > 0
                label, _score, _second = classify(_fit_glyph(glyph), bank)
                read.append(label)
            if all(label is not None and label.isdigit() and label != "0" for label in read):
                amount = 2.0 * int(read[0]) / int(read[1])
        found.append((tip_x, amount))
    return found


def _runs(row: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))) == 1))


def _longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _top_bar(glyph: _Glyph) -> bool:
    """字顶上（上面四分之一）有一道横：最长的一段墨占字宽一半以上。7 有，斜线没有。"""
    rows = glyph.mask[: max(1, glyph.mask.shape[0] // 4)]
    return any(_longest_run(row) >= 0.5 * glyph.width for row in rows)
