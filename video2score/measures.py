"""把一帧切成完整的小节。只留两侧都有小节线的，切残的边缘小节等它完整出现再收。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from video2score.frames import Frame, inverted, score_box
from video2score.staff import System, chroma, find_systems

# 小节上下各留几个线距：上面是小节号、高音和演奏记号，下面是 TAB 下方的节奏符干。
PAD_ABOVE = 6.0
PAD_BELOW = 4.5
# 小节号所在的一块：小节线左边、右边、上面各几个线距。
LABEL_LEFT = 2.5
LABEL_RIGHT = 5.5
LABEL_ABOVE = 3.0
# 下沿停在第一条谱线上方一点。GP 的小节号几乎贴着谱线。
LABEL_BELOW = 0.15
# 行首小节的号：谱号、调号、拍号右端往左、往右各几个线距。
HEAD_LABEL_LEFT = 1.0
HEAD_LABEL_RIGHT = 4.5
# 整页模式下，一行里细横线占纸宽这么多就算谱线。
_PAGE_COVER = 0.2
# 签名图的线距。要能分出 TAB 上的 3 和 5、符头差半个线距；再细就会被压缩噪点拖累。
_SIGNATURE_SPACING = 8.0


@dataclass
class Measure:
    """一帧里的一个完整小节。坐标都是整帧坐标。"""

    frame: int
    seconds: float
    row: int
    left: int
    right: int
    top: float
    spacing: float
    kind: str
    layout: tuple[tuple[int, float], ...]
    lines: tuple[float, ...]
    darkness: np.ndarray = field(repr=False)
    valid: tuple[int, int] = (0, 0)
    signature: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, 1), bool))
    sig_rows: tuple[int, int] = (0, 0)
    head: np.ndarray | None = field(repr=False, default=None)
    label: np.ndarray | None = field(repr=False, default=None)
    label_valid: np.ndarray | None = field(repr=False, default=None)
    # 谱行第一小节才有：谱号、调号、拍号后面那一块。行首小节的号印在这儿，不在小节线上方。
    head_label: np.ndarray | None = field(repr=False, default=None)
    head_label_valid: np.ndarray | None = field(repr=False, default=None)
    # 彩色像素（光标、正在播放时变色的音符）在 darkness 里记成纸；这里存它们按亮度算的深浅（至少 1），别处是 0。
    # 合成时彩色像素不算这一帧看到了，交给别的帧；每帧都是彩色的地方才用这里的值。
    tinted: np.ndarray | None = field(repr=False, default=None)
    # 谱行第一小节才有：它左边那一段（竖排的乐器名、TAB 左边的调弦字母）。
    # 和 darkness 同样的行，宽 PREFIX_WIDTH 个线距，右端贴着 darkness 的左端。别的小节是 None。
    prefix: np.ndarray | None = field(repr=False, default=None)
    # 追踪时读出的小节号和把握（track.Tracker 填写；读不出为 None）。
    number: int | None = None
    number_score: float = 0.0

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def origin(self) -> tuple[int, int]:
        """darkness 左上角在整帧里的位置。"""
        return self.left - _MARGIN, int(round(self.top - PAD_ABOVE * self.spacing))


# 小节线左右各多留几列，拼版时两边的线才不会被切掉。
_MARGIN = 3
# 谱行开头左边截多宽（线距）：乐器名、调弦字母在这里面。
PREFIX_WIDTH = 10.0


@dataclass
class FrameMeasures:
    """一帧里找到的谱行和完整小节。rows 是每行最上面那条线的 y。"""

    frame: Frame
    box: tuple[int, int, int, int]
    rows: list[float]
    spacing: float
    measures: list[Measure]


def frame_measures(
    frame: Frame,
    ratios: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None),
    scale: float = 1.0,
    page: bool = False,
) -> FrameMeasures | None:
    """scale > 1：谱面太小，找到谱面框以后把框里放大这么多倍再找（见 survey）。之后的坐标都是放大后的整帧坐标。
    page：整页视图，谱面框框整张白纸。"""
    box = score_box(frame.rgb, ratios, page)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(frame.rgb[y0:y1, x0:x1])
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        x0, y0 = int(round(x0 * scale)), int(round(y0 * scale))
        x1, y1 = x0 + crop.shape[1], y0 + crop.shape[0]
        box = (x0, y0, x1, y1)
    # 整页视图的谱面框是整张纸：最后一行常常没排满（只剩几个小节），谱线只占纸宽的三四成。
    systems = find_systems(crop, _PAGE_COVER if page else 0.40)
    if page and len(systems) > 1:
        # 整页视图每一行谱的结构都一样（比如五线谱 + TAB）。最上、最下那行线数比别的行少，是被画面边沿切掉了一截
        # （只露出五线谱或只露出 TAB，符干还会被当成小节线），这一帧不收，等它整行出现。
        lines = [sum(staff.count for staff in system.staves) for system in systems]
        ends = {0, len(systems) - 1}
        systems = [system for index, system in enumerate(systems) if index not in ends or lines[index] == max(lines)]
    if not systems:
        return None
    darkness, tinted = darkness_map(crop, with_tinted=True)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    measures: list[Measure] = []
    for row, system in enumerate(systems):
        upper = (systems[row - 1].bottom + system.top) / 2.0 if row > 0 else 0.0
        lower = (system.bottom + systems[row + 1].top) / 2.0 if row + 1 < len(systems) else float(y1 - y0)
        upper, lower = _band_edges(gray, system, upper, lower)
        measures.extend(
            _system_measures(frame, system, box, upper + y0, lower + y0, darkness, row, tinted)
        )
    spacing = float(np.median([system.spacing for system in systems]))
    return FrameMeasures(
        frame=frame,
        box=box,
        rows=[system.top + y0 for system in systems],
        spacing=spacing,
        measures=measures,
    )


class MeasureFinder:
    """逐帧找小节，并记住谱面是白底黑字还是黑底白字。

    先按上一次成功的方向找；找不到再反过来试一次。只有转场、片头的帧会两边都试。
    """

    def __init__(
        self,
        ratios: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None),
        scale: float = 1.0,
        page: bool = False,
    ) -> None:
        self.ratios = ratios
        self.scale = scale
        self.page = page
        self._order = [False, True]
        self.found = {False: 0, True: 0}

    def __call__(self, frame: Frame) -> FrameMeasures | None:
        for invert in self._order:
            found = frame_measures(inverted(frame) if invert else frame, self.ratios, self.scale, self.page)
            if found is not None and found.measures:
                if invert != self._order[0]:
                    self._order.reverse()
                self.found[invert] += 1
                return found
        return None

    @property
    def polarity(self) -> str:
        return "dark" if self.found[True] > self.found[False] else "light"


def _band_edges(gray: np.ndarray, system: System, upper: float, lower: float) -> tuple[float, float]:
    """谱带的上下沿：谱线外侧第一条横贯画面的明暗分界。小节图不越过它。

    谱带是贴在画面上的一条浅色半透明底，外面是插画。越过边沿收进来的插画，
    每次都在同一位置（谱带按固定格子跳），投票也去不掉，只能在这里挡住。
    """
    spacing = system.spacing
    left, right = max(0, system.left), min(gray.shape[1], system.right)
    if right - left < 64:
        return upper, lower
    # 只算上下限之间那几行（整帧算一遍差分很慢，每帧每行谱都要算）。edge[i] 是第 r0+i 行到下一行的变化。
    r0 = max(0, int(upper) - 1)
    r1 = min(gray.shape[0], int(lower) + 2)
    strip = gray[r0:r1, left:right].astype(np.int16)
    step = np.diff(strip, axis=0)
    # 同一行上大多数列都朝同一个方向变亮或变暗，才是横贯的边，不是音符、数字或插画线条。
    edge = np.maximum((step > 6).mean(axis=1), (step < -6).mean(axis=1)) >= 0.5
    rows = gray.shape[0] - 1
    start = int(round(system.top - 1.2 * spacing)) - 1
    stop = max(int(upper), 0)
    for y in range(start, stop, -1):
        if 0 <= y < rows and 0 <= y - r0 < len(edge) and edge[y - r0]:
            upper = float(y + 1)
            break
    start = int(round(system.bottom + 1.2 * spacing))
    stop = min(int(lower), rows)
    for y in range(max(start, 0), stop):
        if 0 <= y - r0 < len(edge) and edge[y - r0]:
            lower = float(y)
            break
    return upper, lower


def _system_measures(
    frame: Frame,
    system: System,
    box: tuple[int, int, int, int],
    upper: float,
    lower: float,
    darkness: np.ndarray,
    row: int,
    tinted: np.ndarray | None = None,
) -> list[Measure]:
    """darkness 只覆盖谱面框 box，坐标要减去框的左上角。"""
    dx, dy, box_right, box_bottom = box
    spacing = system.spacing
    min_width = max(48, int(round(spacing * 4.0)))
    top = system.top + dy
    lines = tuple(line + dy for staff in system.staves for line in staff.lines)
    height = int(round((PAD_ABOVE + PAD_BELOW) * spacing + (system.bottom - system.top)))
    found: list[Measure] = []
    for left, right in zip(system.bars, system.bars[1:]):
        if right - left < min_width:
            continue
        measure = Measure(
            frame=frame.index,
            seconds=frame.seconds,
            row=row,
            left=left + dx,
            right=right + dx,
            top=top,
            spacing=spacing,
            kind=system.kind,
            layout=system.layout,
            lines=tuple(line - top for line in lines),
            darkness=np.zeros((1, 1), np.uint8),
        )
        ox, oy = measure.origin
        width = measure.width + 2 * _MARGIN + 1
        patch = np.zeros((height, width), np.uint8)
        # 超出画面或伸进相邻谱行的部分不收，记在 valid 里，合成时不参与投票。
        lo = max(oy, int(np.ceil(upper)), dy)
        hi = min(oy + height, int(np.floor(lower)), box_bottom)
        cx0 = max(ox, dx)
        cx1 = min(ox + width, box_right)
        if hi - lo < spacing * 3 or cx1 - cx0 < width - 2:
            continue
        patch[lo - oy : hi - oy, cx0 - ox : cx1 - ox] = darkness[lo - dy : hi - dy, cx0 - dx : cx1 - dx]
        measure.darkness = patch
        if not found:
            # 谱行第一小节（开头那段谱线太短、不算小节时，是第二段）左边：乐器名、调弦字母。
            measure.prefix = _prefix_patch(darkness, box, (ox, oy), (lo, hi), height, spacing)
        if tinted is not None:
            colored = np.zeros((height, width), np.uint8)
            colored[lo - oy : hi - oy, cx0 - ox : cx1 - ox] = tinted[lo - dy : hi - dy, cx0 - dx : cx1 - dx]
            measure.tinted = colored
        measure.valid = (lo - oy, hi - oy)
        measure.signature, measure.sig_rows = _signature(patch, measure)
        found_label = _label_patch(darkness, measure, box)
        if found_label is not None:
            measure.label, measure.label_valid = found_label
        if not found and _lines_start_at(darkness, measure, box):
            end = _header_end(patch, measure)
            if end is not None:
                # 右边不能伸到下一小节的号上。
                right_edge = min(end + HEAD_LABEL_RIGHT * spacing, measure.right - 1.0 * spacing)
                span = (int(round(end - HEAD_LABEL_LEFT * spacing)), int(round(right_edge)))
                found_label = _label_patch(darkness, measure, box, span) if span[1] - span[0] >= 2.0 * spacing else None
                if found_label is not None:
                    measure.head_label, measure.head_label_valid = found_label
        measure.head = _head(measure)
        found.append(measure)
    return found


def _prefix_patch(
    darkness: np.ndarray,
    box: tuple[int, int, int, int],
    origin: tuple[int, int],
    rows: tuple[int, int],
    height: int,
    spacing: float,
) -> np.ndarray:
    """谱行开头左边那一段：小节图左端（整帧坐标 origin）往左 PREFIX_WIDTH 个线距，行和小节图对齐，截到谱面框边为止。"""
    dx, dy = box[0], box[1]
    ox, oy = origin
    lo, hi = rows
    width = int(round(PREFIX_WIDTH * spacing))
    patch = np.zeros((height, width), np.uint8)
    x0, x1 = max(ox - width, dx), ox
    if x1 > x0:
        patch[lo - oy : hi - oy, width - (x1 - x0) :] = darkness[lo - dy : hi - dy, x0 - dx : x1 - dx]
    return patch


def darkness_map(rgb: np.ndarray, with_tinted: bool = False):
    """每个像素有多"像墨"：0 是纸，255 是谱线那么黑。彩色像素（光标、彩色高亮）记 0。

    深浅按局部的纸色和谱线色拉伸。谱带后面的插画比纸暗，但比墨浅得多，落到中间值，
    后面多帧投票和收尾阈值会把它去掉。

    with_tinted 时另外返回彩色像素本来的深浅（其余是 0）：正在播放的小节里音符会变色，
    记成纸就等于这一帧里音符没了。
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # 纸色：大窗口里的亮部。墨色：固定按接近黑来算，视频压缩后黑线在 20~70 之间。
    paper = cv2.dilate(gray, np.ones((31, 31), np.uint8))
    paper = cv2.blur(paper, (61, 61))
    ink_level = 40.0
    span = cv2.max(paper - ink_level, 40.0)
    value = cv2.divide(paper - gray, span)
    value = np.clip(value * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = chroma(rgb) > 70
    # 彩色像素至少记 1，好和不是彩色的像素（0）分开：彩色的纸（高亮框底色）这一帧也不算看到。
    tinted = np.where(colored, np.maximum(value, 1), 0).astype(np.uint8)
    value[colored] = 0
    return (value, tinted) if with_tinted else value


def _signature(patch: np.ndarray, measure: Measure) -> tuple[np.ndarray, tuple[int, int]]:
    """小节内部的墨迹缩略图，去掉谱线和两边的小节线。用来判断两帧里是不是同一个小节。

    缩略图总是覆盖谱线上下各三个线距。被画面边缘截掉的行记在返回的行范围外，比较时跳过。
    """
    spacing = measure.spacing
    top = int(round(PAD_ABOVE * spacing))
    lo = int(round(top - 3.0 * spacing))
    bottom = top + (measure.lines[-1] if measure.lines else 0)
    hi = int(round(bottom + 3.0 * spacing))
    ink = patch[lo:hi] >= 150
    band = max(1, int(round(spacing * 0.12)))
    for line in measure.lines:
        row = int(round(top + line)) - lo
        ink[max(0, row - band) : row + band + 1] = False
    edge = _MARGIN + max(3, int(round(spacing * 0.3)))
    ink = ink[:, edge : ink.shape[1] - edge]
    if ink.size == 0:
        return np.zeros((1, 1), bool), (0, 0)
    scale = _SIGNATURE_SPACING / spacing
    size = (max(1, int(round(ink.shape[1] * scale))), max(1, int(round(ink.shape[0] * scale))))
    small = cv2.resize(ink.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.2
    first = int(np.ceil(max(0, measure.valid[0] - lo) * scale))
    last = int(np.floor(max(0, min(hi, measure.valid[1]) - lo) * scale))
    return small, (min(first, small.shape[0]), min(last, small.shape[0]))


def _label_patch(
    darkness: np.ndarray,
    measure: Measure,
    box: tuple[int, int, int, int],
    span: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """小节线上方、左右各伸出几个线距的一块：小节号印在这里，常常骑在小节线上。span 给出别的左右范围（整帧坐标）。

    返回 (这一块, 哪些像素在画面里)；被谱面框截掉时返回 None。
    """
    spacing = measure.spacing
    x0 = int(round(measure.left - LABEL_LEFT * spacing)) if span is None else span[0]
    x1 = int(round(measure.left + LABEL_RIGHT * spacing)) if span is None else span[1]
    y0 = int(round(measure.top - LABEL_ABOVE * spacing))
    y1 = int(round(measure.top - LABEL_BELOW * spacing))
    bx0, by0, bx1, by1 = box
    if y1 > by1:
        return None
    # 被谱面框截掉的部分补 0，并记进 label_valid。大小保持一致，才能逐像素比。
    cy0, cx0, cx1 = max(y0, by0), max(x0, bx0), min(x1, bx1)
    if y1 - cy0 < 1.6 * spacing or cx1 - cx0 < 1.5 * spacing:
        return None
    label = np.zeros((y1 - y0, x1 - x0), np.uint8)
    label[cy0 - y0 :, cx0 - x0 : cx1 - x0] = darkness[cy0 - by0 : y1 - by0, cx0 - bx0 : cx1 - bx0]
    valid = np.zeros(label.shape, bool)
    valid[cy0 - y0 :, cx0 - x0 : cx1 - x0] = True
    return label, valid


def _lines_start_at(darkness: np.ndarray, measure: Measure, box: tuple[int, int, int, int]) -> bool:
    """谱线从这一小节的左小节线才开始（谱行的开头）：小节线左边 0.5~2 个线距里，谱线那几行基本没墨。
    横向滚动时画面最左边那一格的左边谱线还连着，不是谱行开头（那儿开头的一摞和弦数字不是谱号）。"""
    dx, dy, _box_right, _box_bottom = box
    spacing = measure.spacing
    x0 = int(round(measure.left - 2.0 * spacing)) - dx
    x1 = int(round(measure.left - 0.5 * spacing)) - dx
    if x1 <= 0:
        return True
    x0 = max(0, x0)
    rows = [int(round(measure.top + line)) - dy for line in measure.lines]
    rows = [row for row in rows if 0 <= row < darkness.shape[0]]
    if not rows:
        return False
    covered = np.mean([(darkness[max(0, row - 1) : row + 2, x0:x1] >= 128).any(axis=0).mean() for row in rows])
    return covered < 0.3


def _header_end(patch: np.ndarray, measure: Measure) -> int | None:
    """谱行第一小节开头的谱号、调号、拍号的右端（整帧的 x）；开头没有谱号返回 None。

    和 header._header_end 一样按列找符号段：第一段贴着左边小节线、高过 2 个线距是谱号；后面挨着的窄段是调号；
    又高又粗（上下两个粗体数字）的是拍号，算进去以后就停。
    """
    from video2score.header import _bold, _height, _segments

    spacing = measure.spacing
    ox, oy = measure.origin
    count = int(measure.layout[0][0]) if measure.layout else 5
    ys = [measure.top + line - oy for line in measure.lines[:count]]
    y0 = max(0, int(ys[0] - 1.0 * spacing))
    y1 = min(patch.shape[0], int(ys[-1] + 1.0 * spacing) + 1)
    if y1 - y0 < 3:
        return None
    ink = (patch[y0:y1] >= 128).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(5, int(round(1.5 * spacing)))), np.uint8))
    symbols = cv2.subtract(ink, lines)
    start = measure.left - ox + max(2, int(round(0.25 * spacing)))
    stop = measure.right - ox
    if stop - start < 4:
        return None
    segments = _segments(symbols[:, start:stop].sum(axis=0) > 0, max(1, int(round(0.3 * spacing))))
    if not segments:
        return None
    heights = [_height(symbols[:, start + a : start + b]) for a, b in segments]
    if segments[0][0] > 1.5 * spacing or heights[0] < 2.0 * spacing:
        return None
    end = segments[0][1]
    for (a, b), height in zip(segments[1:], heights[1:]):
        # 调号和拍号挨得紧时会连成一段（六个升号加拍号有十个线距宽）。
        if a - end > 1.5 * spacing or b - a > 12.0 * spacing:
            break
        end = b
        if _bold(symbols[:, start + a : start + b], spacing) > 0.3 and height >= 3.3 * spacing:
            break
    return ox + start + end


def _head(measure: Measure) -> np.ndarray | None:
    """小节号的笔画。

    很多谱一连几小节音符完全一样，只有小节号不同。这一块单独比，才分得开。
    小节号常是浅灰细字，按细笔画取（比周围暗一截的窄线条），背景插画的大片明暗不算。
    """
    if measure.label is None:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    strokes = cv2.subtract(measure.label, cv2.morphologyEx(measure.label, cv2.MORPH_OPEN, kernel))
    return (strokes >= 60) & measure.label_valid


def labels_agree(left: Measure, right: Measure) -> bool:
    """小节号那一块没有明显冲突。背景插画的细线也会混进来，所以只在比较几种可能时用。"""
    return not _heads_differ(left, right)


def _heads_differ(left: Measure, right: Measure) -> bool:
    """两边小节号那一块都有字、而字对不上。一边没字或被截掉时不下结论。"""
    found = _heads_distance(left, right)
    # 22 和 26 只有一位不同，一半的笔画是重合的；同一个号在两帧里几乎完全重合。
    return found is not None and found[2] >= 8 and found[0] > 0.08


def _heads_distance(
    left: Measure,
    right: Measure,
    reach: int | None = None,
) -> tuple[float, int, int] | None:
    """(对不上的比例, 两边笔画总数, 对不上的像素)，取左右挪动后最好的一次。没法比时返回 None。"""
    a, b = left.head, right.head
    if a is None or b is None or left.label_valid is None or right.label_valid is None:
        return None
    # 线距差零点几像素时两块会差一行一列，裁成一样大。
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    valid_a = left.label_valid[:height, :width]
    valid_b = right.label_valid[:height, :width]
    a = a[:height, :width]
    b = b[:height, :width]
    # 小节线偶尔会差几像素（动画、双线），号也跟着错开。左右挪一挪取最好的。
    if reach is None:
        reach = max(2, int(round(left.spacing * 0.5)))
    # 只留有笔画的那一块（四周多留挪动的距离和膨胀的一像素）：没墨的地方怎么挪都不影响计数，块小了快很多。
    rows = np.flatnonzero((a | b).any(axis=1))
    columns = np.flatnonzero((a | b).any(axis=0))
    if rows.size and columns.size:
        r0, r1 = max(0, int(rows[0]) - 1), int(rows[-1]) + 2
        c0, c1 = max(0, int(columns[0]) - reach - 1), int(columns[-1]) + reach + 2
        a, b = a[r0:r1, c0:c1], b[r0:r1, c0:c1]
        valid_a, valid_b = valid_a[r0:r1, c0:c1], valid_b[r0:r1, c0:c1]
    kernel = np.ones((3, 3), np.uint8)
    best: tuple[float, int, int] | None = None
    for shift in range(-reach, reach + 1):
        moved_b = np.roll(b, shift, axis=1)
        common = valid_a & np.roll(valid_b, shift, axis=1)
        if shift > 0:
            common[:, :shift] = False
        elif shift < 0:
            common[:, shift:] = False
        one = a & common
        two = moved_b & common
        total = int(one.sum()) + int(two.sum())
        if int(one.sum()) < 15 or int(two.sum()) < 15:
            return None
        # 数字只有十来个像素高，只容一像素抖动，否则 5 和 6 互相都盖得住。
        grown_one = cv2.dilate(one.astype(np.uint8), kernel) > 0
        grown_two = cv2.dilate(two.astype(np.uint8), kernel) > 0
        missing = int((one & ~grown_two).sum()) + int((two & ~grown_one).sum())
        found = (missing / total, total, missing)
        if best is None or found < best:
            best = found
    return best


def same_layout(left: tuple, right: tuple) -> bool:
    """两次观测的谱线分组一样：每组线数、谱类相同，各组相对第一条线的偏移差不到 0.3 个线距。
    偏移按 0.1 个线距四舍五入，正好卡在半档上的会在两档之间来回跳（见过 TAB 在 10.8 和 10.9 之间跳，
    要求完全相等时一整屏小节帧帧都对不上，被当成残影丢掉）。"""
    return len(left) == len(right) and all(
        a[0] == b[0] and a[2:] == b[2:] and abs(a[1] - b[1]) < 0.3 for a, b in zip(left, right)
    )


def same_measure(left: Measure, right: Measure) -> bool:
    """同一张谱上的同一个小节：谱类一样、宽度差不多、墨迹缩略图对得上。

    一连几小节音符相同时这里分不开，要再看 labels_agree。
    """
    if not same_layout(left.layout, right.layout):
        return False
    if abs(left.width - right.width) > max(6.0, 0.05 * max(left.width, right.width)):
        return False
    if abs(left.spacing - right.spacing) > max(0.8, 0.06 * left.spacing):
        return False
    a, b = _common_rows(left, right)
    if a is None or b is None:
        return False
    return signature_distance(a, b) <= 0.16


def loosely_same(left: Measure, right: Measure) -> bool:
    """位置已经对上时用：长相大体一致即可。"""
    if not same_layout(left.layout, right.layout):
        return False
    if abs(left.spacing - right.spacing) > max(0.8, 0.06 * left.spacing):
        return False
    a, b = _common_rows(left, right)
    if a is None or b is None:
        return False
    return signature_distance(a, b) <= 0.40


def _common_rows(left: Measure, right: Measure) -> tuple[np.ndarray | None, np.ndarray | None]:
    """两次观测都完整可见的那几行。至少要盖住谱线那一段。"""
    a, b = left.signature, right.signature
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    first = max(left.sig_rows[0], right.sig_rows[0])
    last = min(left.sig_rows[1], right.sig_rows[1], a.shape[0])
    needed = int(round(_SIGNATURE_SPACING * 3.0))
    if first > needed or last < a.shape[0] - needed:
        return None, None
    return a[first:last], b[first:last]


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """两张缩略图互相覆盖不到的墨迹比例。膨胀一格，容许一像素抖动。"""
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    total = int(a.sum()) + int(b.sum())
    if total < 12:
        # 两边都几乎是空小节（整小节休止）。一边空一边满时 total 不会这么小。
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    grown_a = cv2.dilate(a.astype(np.uint8), kernel) > 0
    grown_b = cv2.dilate(b.astype(np.uint8), kernel) > 0
    missing = int((a & ~grown_b).sum()) + int((b & ~grown_a).sum())
    return missing / total
