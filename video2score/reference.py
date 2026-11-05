"""读 Guitar Pro 导出的 PDF 乐谱，当作比对用的标准答案。

GP 导出的 PDF 是矢量的：
- 符头、休止符、符尾、附点、拍号用 Bravura 音乐字体（SMuFL 码位），是文字；
- TAB 上的品位是普通文字（括号表示延音）；
- 谱线是一条长路径，小节线是细长的填充矩形，符干是单根竖线，符杠是细长的填充四边形；
- 连音的 3 是斜体 Times 文字；合并休止的小节数是大号的拍号数字，印在谱表上方。

只读这些就能还原出每小节的每个音：时值、是不是休止、TAB 上的弦和品。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

# SMuFL 码位。
_HEADS = {0xE0A2: Fraction(1), 0xE0A3: Fraction(1, 2), 0xE0A4: Fraction(1, 4), 0xE0A9: Fraction(1, 4)}
_RESTS = {
    0xE4E3: Fraction(1),
    0xE4E4: Fraction(1, 2),
    0xE4E5: Fraction(1, 4),
    0xE4E6: Fraction(1, 8),
    0xE4E7: Fraction(1, 16),
    0xE4E8: Fraction(1, 32),
}
_FLAGS = {0xE240: 1, 0xE241: 1, 0xE242: 2, 0xE243: 2, 0xE244: 3, 0xE245: 3}
_DOT = 0xE1E7
_TIME_DIGITS = range(0xE080, 0xE08A)
_CLEFS = {0xE050, 0xE052, 0xE062}  # 高音谱号、低八度高音谱号、低音谱号
_FLAT, _NATURAL, _SHARP = 0xE260, 0xE261, 0xE262
# 谱面上下的演奏记号（朝上、朝下各一个码位）。
_ARTICULATIONS = {
    0xE4A0: "accent", 0xE4A1: "accent",
    0xE4A2: "staccato", 0xE4A3: "staccato",
    0xE4AC: "strong-accent", 0xE4AD: "strong-accent",
    0xE4C0: "fermata", 0xE4C1: "fermata",
}
_REPEAT_DOT = 0xE044
_VIBRATO = 0xEAB2
_BEND_ARROW = 0xEB78
_TEMPO_NOTE = 0xE1D5  # 速度前面的四分音符
_STRING_1 = 0xE834  # 圈起来的弦号 1，往后依次是 2、3……
_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_FRACTIONS = {"¼": 0.25, "½": 0.5, "¾": 0.75}


@dataclass
class RefFret:
    string: int
    fret: int | None  # None 是闷音 X
    paren: bool = False


@dataclass
class RefEvent:
    x: float
    duration: Fraction
    rest: bool = False
    dots: int = 0
    tuplet: Fraction | None = None
    frets: list[RefFret] = field(default_factory=list)
    # 演奏技巧：slide-start、slide-stop、scoop、doit、falloff、hammer、pull、staccato、tie-start、tie-stop。
    marks: set[str] = field(default_factory=set)
    # 紧挨在它前面的装饰音个数（装饰音本身不算事件，不占时值）。
    graces: int = 0
    # 读 PDF 时装饰音的品位，一个装饰音一个。
    grace_frets: list[RefFret] = field(default_factory=list)
    grace_x: float = 0.0  # 最后一个装饰音的横坐标
    grace_marks: set[str] = field(default_factory=set)  # 装饰音上的技巧（装饰音击弦、勾弦到这个音）


@dataclass
class RefMeasure:
    number: int
    page: int
    events: list[RefEvent]
    meter: Fraction | None = None
    # 合并休止：这一格代表几个小节。
    count: int = 1
    # 调号（升号数为正、降号数为负）和速度（每分钟四分音符数），只在印了的小节有。
    key: int | None = None
    tempo: float | None = None
    # TAB 有几根弦（读 PDF 时填；没有 TAB 是 0）。
    strings: int = 0
    # 调弦（MIDI 音高，1 弦在前），读 PDF 时只在第一小节填，谱头没印就是 None。
    tuning: tuple[int, ...] | None = None
    # 反复记号（读 PDF 时填）：左边是正向反复线；右边是反向反复线时，这一段一共演几遍（0 是没有）。
    repeat_start: bool = False
    repeat_times: int = 0
    # 小节在页面上的方框 (x0, y0, x1, y1)，PDF 坐标（读 PDF 时填）：矢量作业按它裁小节图。
    box: tuple[float, float, float, float] | None = None

    @property
    def total(self) -> Fraction:
        return sum((event.duration for event in self.events), Fraction(0))


@dataclass
class _Glyph:
    code: int
    text: str
    font: str
    size: float
    x0: float
    y0: float
    x1: float
    y1: float
    ox: float
    oy: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class _Staff:
    lines: list[float]
    x0: float
    x1: float

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def bottom(self) -> float:
        return self.lines[-1]

    @property
    def gap(self) -> float:
        return (self.lines[-1] - self.lines[0]) / (len(self.lines) - 1)


@dataclass
class _System:
    staff: _Staff | None
    tab: _Staff | None

    @property
    def top(self) -> float:
        return (self.staff or self.tab).top

    @property
    def bottom(self) -> float:
        return (self.tab or self.staff).bottom

    @property
    def x0(self) -> float:
        return (self.staff or self.tab).x0

    @property
    def x1(self) -> float:
        return (self.staff or self.tab).x1


def read_reference(path: Path) -> list[RefMeasure]:
    """标准答案：GP 导出的 PDF，或者 MusicXML（.musicxml / .xml，见 refxml.py）。"""
    if Path(path).suffix.lower() in (".musicxml", ".xml"):
        from video2score.refxml import read_musicxml

        return read_musicxml(Path(path))
    import pymupdf

    measures: list[RefMeasure] = []
    carry: list[RefEvent] = []  # 滑到下一行的滑音起点
    tuning = None
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            measures.extend(_read_page(page, page_number, measures, carry))
            if page_number == 1 and measures:
                tuning = _tuning(_glyphs(page), max(measure.strings for measure in measures))
    if measures:
        measures[0].tuning = tuning
    # 延音的后半个音在 TAB 上常常不写数字：品位抄前一个音的。
    previous: RefEvent | None = None
    for measure in measures:
        for event in measure.events:
            if event.rest:
                previous = None
                continue
            if not event.frets and previous is not None:
                event.frets = [RefFret(fret.string, fret.fret, True) for fret in previous.frets]
            previous = event
    # 合并休止：一格里没有音、下一格的号跳了好几个，这一格就代表中间这些小节。
    for measure, following in zip(measures, measures[1:]):
        gap = following.number - measure.number
        if gap > 1 and all(event.rest for event in measure.events):
            measure.count = gap
            measure.events = []
    return _played(measures)


def _played(measures: list[RefMeasure]) -> list[RefMeasure]:
    """反复记号按演奏顺序展开，从第一小节的号起连续编号（和 MusicXML 答案、视频里展开画的谱一样）。

    shortcut: 不认几房跳跃和 D.S./D.C.，遇到有这些记号的谱再加。"""
    if not any(measure.repeat_times for measure in measures):
        return measures
    import copy

    order: list[RefMeasure] = []
    start = 0
    jumped: dict[int, int] = {}
    seen: set[int] = set()
    index = 0
    while index < len(measures):
        measure = measures[index]
        if measure.repeat_start:
            start = index
        order.append(copy.deepcopy(measure) if index in seen else measure)
        seen.add(index)
        if measure.repeat_times and jumped.get(index, 0) < measure.repeat_times - 1:
            jumped[index] = jumped.get(index, 0) + 1
            index = start
            continue
        if measure.repeat_times:
            start = index + 1
        index += 1
    number = order[0].number
    for measure in order:
        measure.number = number
        number += measure.count
    return order


def _read_page(page, page_number: int, before: list[RefMeasure], carry: list[RefEvent]) -> list[RefMeasure]:
    glyphs = _glyphs(page)
    drawings = page.get_drawings()
    systems = _systems(drawings)
    stems, beams, bars = _strokes(drawings)
    slides = _slide_lines(drawings)
    # 推弦曲线：描边的三次曲线，从推弦的那个音起头，到箭头结束。
    curves = [(item[1], item[-1]) for drawing in drawings if drawing["type"] == "s" for item in drawing["items"] if item[0] == "c"]
    out: list[RefMeasure] = []
    ceiling = 0.0  # 上一行谱的底
    for system in systems:
        edges = _measure_edges(system, bars)
        row: list[tuple[float, RefMeasure]] = []
        for left, right in zip(edges, edges[1:]):
            if right - left < 8:
                continue
            previous = out[-1] if out else (before[-1] if before else None)
            measure = _read_measure(system, left, right, glyphs, stems, beams, page_number, previous)
            if system.staff is not None and not measure.events and measure.count == 1:
                continue  # 行首谱号、调号和正向反复线之间的空当
            out.append(measure)
            row.append((left, measure))
        if row:
            _system_marks(system, row, glyphs, slides, curves, carry, ceiling)
        ceiling = system.bottom
    return out


def _glyphs(page) -> list[_Glyph]:
    found: list[_Glyph] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for char in span["chars"]:
                    text = char["c"]
                    if not text.strip():
                        continue
                    x0, y0, x1, y1 = char["bbox"]
                    ox, oy = char["origin"]
                    found.append(_Glyph(ord(text), text, span["font"], float(span["size"]), x0, y0, x1, y1, ox, oy))
    return found


def _systems(drawings) -> list[_System]:
    """谱线：长横线按 y 归并，等距的 5 条是五线谱，4~7 条间距更大的是 TAB。"""
    rows: dict[float, list[float]] = {}
    for drawing in drawings:
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            if abs(a.y - b.y) > 0.01 or abs(a.x - b.x) < 3:
                continue
            key = round(a.y, 1)
            span = rows.setdefault(key, [min(a.x, b.x), max(a.x, b.x), 0.0])
            span[0], span[1] = min(span[0], a.x, b.x), max(span[1], a.x, b.x)
            span[2] += abs(a.x - b.x)
    # 加线很短，和谱线等距，混进来会让五线谱变成六七条线：只要累计够长的线。
    ys = sorted(y for y, span in rows.items() if span[2] >= 100)
    # 从每条线往下找间距一致（3~9 点）的一串，够 4 条就是一组谱线。
    groups: list[list[float]] = []
    index = 0
    while index < len(ys):
        best: list[float] = [ys[index]]
        for following in ys[index + 1 :]:
            gap = following - ys[index]
            if gap > 9:
                break
            if gap < 3:
                continue
            run = [ys[index], following]
            for y in ys:
                if y > run[-1] and abs((y - run[-1]) - gap) < 0.3:
                    run.append(y)
                elif y > run[-1] + gap + 0.3:
                    break
            if len(run) > len(best):
                best = run
        if len(best) > 5 and best[1] - best[0] < 5.5:
            # 五线谱上下还有一条等距的长线（很多加线连成一片）：取最长的连续 5 条。
            start = max(range(len(best) - 4), key=lambda i: sum(rows[y][2] for y in best[i : i + 5]))
            best = best[start : start + 5]
        if len(best) >= 4:
            groups.append(best)
            index = ys.index(best[-1]) + 1
        else:
            index += 1
    staves = [
        _Staff(group, min(rows[y][0] for y in group), max(rows[y][1] for y in group))
        for group in groups
        if len(group) >= 4
    ]
    systems: list[_System] = []
    index = 0
    while index < len(staves):
        staff = staves[index]
        if len(staff.lines) == 5 and staff.gap < 5.5:
            following = staves[index + 1] if index + 1 < len(staves) else None
            if following is not None and following.gap >= 5.5 and following.top - staff.bottom < 60:
                systems.append(_System(staff, following))
                index += 2
                continue
            systems.append(_System(staff, None))
        else:
            systems.append(_System(None, staff))
        index += 1
    return systems


def _strokes(drawings):
    stems: list[tuple[float, float, float]] = []
    beams: list[tuple[float, float, float, float]] = []
    bars: list[tuple[float, float, float]] = []
    for drawing in drawings:
        items = drawing["items"]
        rect = drawing["rect"]
        if drawing["type"] == "s" and len(items) == 1 and items[0][0] == "l":
            a, b = items[0][1], items[0][2]
            if abs(a.x - b.x) < 0.05 and abs(a.y - b.y) > 3:
                # 虚线是虚小节线，实线是符干。
                dashed = (drawing.get("dashes") or "[] 0").replace(" ", "") != "[]0"
                (bars if dashed else stems).append((a.x, min(a.y, b.y), max(a.y, b.y)))
        elif drawing["type"] == "f" and len(items) == 1 and items[0][0] == "re":
            if rect.width < 1.6 and rect.height > 8:
                bars.append(((rect.x0 + rect.x1) / 2.0, rect.y0, rect.y1))
        elif drawing["type"] == "f" and len(items) == 4 and all(item[0] == "l" for item in items):
            points = [item[1] for item in items]
            left = min(points, key=lambda point: point.x)
            right = max(points, key=lambda point: point.x)
            if right.x - left.x > 2:
                # 上沿两个端点：(左 x, 左 y, 右 x, 右 y)。
                top_left = min((p for p in points if abs(p.x - left.x) < 0.01), key=lambda p: p.y)
                top_right = min((p for p in points if abs(p.x - right.x) < 0.01), key=lambda p: p.y)
                beams.append((top_left.x, top_left.y, top_right.x, top_right.y))
    return stems, beams, bars


def _measure_edges(system: _System, bars) -> list[float]:
    # 小节线从谱表顶线画到底线；合并休止横杠两头的短竖线不算。
    xs = sorted(
        x
        for x, y0, y1 in bars
        if system.x0 - 1 <= x <= system.x1 + 1
        and any(
            y0 <= part.top + 0.5 and y1 >= part.bottom - 0.5 for part in (system.staff, system.tab) if part is not None
        )
    )
    edges: list[float] = []
    for x in [system.x0] + xs + [system.x1]:
        if edges and x - edges[-1] < 4:
            # 双小节线、终止线：两条挨着的算一条，取靠右那条作左边界。
            edges[-1] = x
            continue
        edges.append(x)
    return edges


def _read_measure(system, left, right, glyphs, stems, beams, page_number, previous) -> RefMeasure:
    staff, tab = system.staff, system.tab
    top = system.top - 24  # 上加线的高音符头也算进来
    inside = [g for g in glyphs if left - 0.5 <= g.cx <= right + 0.5 and top <= (g.y0 + g.y1) / 2 <= system.bottom + 12]
    number = _number(glyphs, system, left)
    if number is None:
        number = previous.number + previous.count if previous else 1
    measure = RefMeasure(number, page_number, [], strings=len(tab.lines) if tab is not None else 0, box=(left, top, right, system.bottom + 12))
    bravura = [g for g in inside if "Bravura" in g.font]
    if staff is not None:
        # 拍号：谱表里的拍号数字；合并休止的小节数：谱表上方的大号数字。
        time = [g for g in bravura if g.code in _TIME_DIGITS and staff.top - 1 <= g.oy <= staff.bottom + 8]
        above = [g for g in bravura if g.code in _TIME_DIGITS and g.oy < staff.top - 1]
        if time:
            middle = (staff.top + staff.bottom) / 2.0
            upper = "".join(str(g.code - 0xE080) for g in sorted(time, key=lambda g: g.x0) if g.oy <= middle + 0.5)
            lower = "".join(str(g.code - 0xE080) for g in sorted(time, key=lambda g: g.x0) if g.oy > middle + 0.5)
            if upper and lower:
                measure.meter = Fraction(int(upper), int(lower))
        if above:
            measure.count = int("".join(str(g.code - 0xE080) for g in sorted(above, key=lambda g: g.x0)))
        measure.events, graces = _staff_events(staff, bravura, inside, stems, beams, left, right)
        measure.key = _key(staff, bravura, left)
    else:
        graces = []
    frets = _frets(tab, inside) if tab is not None else []
    small = _frets(tab, inside, grace=True) if tab is not None and graces else []
    for event in measure.events:
        if event.rest:
            continue
        event.frets = [fret for x, fret in frets if abs(x - event.x) < 3.5]
    # 装饰音记在它后面那个音上。
    for x in graces:
        owner = next((event for event in measure.events if event.x > x), None)
        if owner is not None:
            owner.graces += 1
            owner.grace_x = x
            owner.grace_frets.extend(fret for fx, fret in small if abs(fx - x) < 3.5)
    if measure.count > 1:
        measure.events = []
    # 反复线：小节线里侧的两个点；反向反复线上方印“3x”时演 3 遍，没印是 2 遍。
    dots = [g.cx for g in bravura if g.code == _REPEAT_DOT]
    measure.repeat_start = any(left < x < left + 6 for x in dots)
    if any(right - 7 < x < right for x in dots):
        times = "".join(g.text for g in sorted(glyphs, key=lambda g: g.x0) if "Bold" in g.font and right - 15 < g.x0 < right + 8 and g.oy < system.top and system.top - g.oy < 25 and (g.text.isdigit() or g.text == "x"))
        measure.repeat_times = int(times[:-1]) if times.endswith("x") and times[:-1].isdigit() else 2
    return measure


def _number(glyphs, system, left) -> int | None:
    """小节号：谱表上方、左边小节线旁边的小号 Times 数字。

    行首小节的左边界在谱号前面，号印在谱号后面，离得较远：起头那一位在范围里就行，后面挨着的位照样接上。
    """
    digits = sorted(
        (
            g
            for g in glyphs
            if g.font.startswith("TimesNewRoman") and "Bold" not in g.font and "Italic" not in g.font
            and g.text.isdigit() and g.size < 8
            and system.top - 10 < g.y1 <= system.top + 0.5
        ),
        key=lambda g: g.x0,
    )
    starts = [g for g in digits if left - 10 <= g.x0 <= left + 60]
    if not starts:
        return None
    chain = [starts[0]]
    for g in digits:
        if g.x0 > chain[-1].x0 and g.x0 - chain[-1].x1 < 1.0 and abs(g.y1 - chain[-1].y1) < 1:
            chain.append(g)
    return int("".join(g.text for g in chain))


def _staff_events(staff, bravura, inside, stems, beams, left, right) -> tuple[list[RefEvent], list[float]]:
    """五线谱上的事件，加上装饰音符头的横坐标（装饰音的符头小一号，不占时值）。"""
    heads = [g for g in bravura if g.code in _HEADS]
    full = max(g.size for g in heads) if heads else 0
    graces = sorted(g.cx for g in heads if g.size < 0.9 * full)
    heads = [g for g in heads if g.size >= 0.9 * full]
    flags = [g for g in bravura if g.code in _FLAGS]
    dots = [g for g in bravura if g.code == _DOT]
    events: list[RefEvent] = []
    used_stems: dict[int, RefEvent] = {}
    for head in sorted(heads, key=lambda g: g.x0):
        y = head.oy
        # 符干贴着符头左边（朝下）或右边（朝上）；音挤在一起时两边都有，取离边最近的一根。
        touching = [
            (min(abs(x - head.x0), abs(x - head.x1)), index)
            for index, (x, y0, y1) in enumerate(stems)
            if head.x0 - 0.8 <= x <= head.x1 + 0.8 and y0 - 2 <= y <= y1 + 2 and left <= x <= right
        ]
        stem_index = min(touching)[1] if touching else None
        if stem_index is not None and stem_index in used_stems:
            continue  # 和弦里的另一个符头
        base = _HEADS[head.code]
        count = 0
        if stem_index is not None and base == Fraction(1, 4):
            x, y0, y1 = stems[stem_index]
            up = abs(y - y1) < abs(y - y0)
            free = y0 if up else y1
            count = _beams_at(beams, x, free)
            if count == 0:
                count = max((_FLAGS[g.code] for g in flags if abs(g.ox - x) < 1.5), default=0)
        duration = base / (2**count)
        dot = sum(1 for g in dots if head.x1 - 0.5 < g.x0 < head.x1 + 8 and abs(g.oy - y) < 3.5)
        duration *= Fraction(3, 2) if dot == 1 else Fraction(7, 4) if dot >= 2 else 1
        event = RefEvent(x=head.cx, duration=duration, dots=min(dot, 2))
        if stem_index is not None:
            used_stems[stem_index] = event
        events.append(event)
    for rest in (g for g in bravura if g.code in _RESTS):
        duration = _RESTS[rest.code]
        dot = sum(1 for g in dots if rest.x1 - 0.5 < g.x0 < rest.x1 + 6 and abs(g.oy - rest.oy) < 6)
        duration *= Fraction(3, 2) if dot else 1
        events.append(RefEvent(x=rest.cx, duration=duration, rest=True, dots=min(dot, 1)))
    events.sort(key=lambda event: event.x)
    # 连音：斜体的 3（5、6），取离它最近的 n 个事件。
    for g in inside:
        if "Italic" in g.font and g.text in "356" and g.size >= 7:
            number = int(g.text)
            normal = 1
            while normal * 2 < number:
                normal *= 2
            candidates = [e for e in events if e.tuplet is None]
            if len(candidates) < number:
                continue
            start = min(
                range(len(candidates) - number + 1),
                key=lambda s: abs(sum(e.x for e in candidates[s : s + number]) / number - g.cx),
            )
            for event in candidates[start : start + number]:
                event.tuplet = Fraction(normal, number)
                event.duration *= event.tuplet
    return events, graces


def _beams_at(beams, x: float, free: float) -> int:
    """横过这根符干、靠近自由端的符杠有几道。"""
    ys = []
    for x0, y0, x1, y1 in beams:
        if not x0 - 0.4 <= x <= x1 + 0.4:
            continue
        y = y0 + (y1 - y0) * ((x - x0) / (x1 - x0) if x1 > x0 else 0)
        if abs(y - free) < 9:
            ys.append(y)
    ys.sort()
    count = 0
    for index, y in enumerate(ys):
        if index == 0 or y - ys[index - 1] > 1.0:
            count += 1
    return count


def _frets(tab: _Staff, inside, grace: bool = False) -> list[tuple[float, RefFret]]:
    """TAB 上的品位：同一行里挨着的字符拼成一个数，括号是延音，X 是闷音。grace=True 时只要小一号的（装饰音的）。"""
    chars = [
        g
        for g in inside
        if "Bravura" not in g.font and not g.font.startswith("TimesNewRoman")
        and tab.top - 3 <= (g.y0 + g.y1) / 2 <= tab.bottom + 3
        and (g.text.isdigit() or g.text in "()xX")
    ]
    if not chars:
        return []
    full = max(g.size for g in chars)
    chars = [g for g in chars if (g.size < 0.9 * full) == grace]
    chars.sort(key=lambda g: (round((g.y0 + g.y1) / 2), g.x0))
    tokens: list[list[_Glyph]] = []
    for g in chars:
        if tokens:
            last = tokens[-1][-1]
            if abs((last.y0 + last.y1) / 2 - (g.y0 + g.y1) / 2) < 1 and g.x0 - last.x1 < 0.3:  # 同一个数的字符首尾相接，挤在一起的两个音也隔着约 0.7 点
                tokens[-1].append(g)
                continue
        tokens.append([g])
    out: list[tuple[float, RefFret]] = []
    for token in tokens:
        text = "".join(g.text for g in token)
        paren = "(" in text
        core = text.strip("()")
        if not core:
            continue
        y = (token[0].y0 + token[0].y1) / 2
        string = 1 + min(range(len(tab.lines)), key=lambda i: abs(tab.lines[i] - y))
        digits = [g for g in token if g.text.isdigit() or g.text in "xX"]
        x = (digits[0].x0 + digits[-1].x1) / 2
        fret = None if core.lower() == "x" else int(core) if core.isdigit() else None
        out.append((x, RefFret(string, fret, paren)))
    return out


def _slide_lines(drawings) -> list[tuple[float, float, float, float]]:
    """滑音斜线：单根描边的斜直线，比符干粗（约 0.68 点）。返回 (左 x, 左 y, 右 x, 右 y)。"""
    lines = []
    for drawing in drawings:
        items = drawing["items"]
        if drawing["type"] != "s" or len(items) != 1 or items[0][0] != "l":
            continue
        if not 0.6 <= (drawing.get("width") or 0) <= 0.8:
            continue
        a, b = sorted((items[0][1], items[0][2]), key=lambda point: point.x)
        if b.x - a.x > 1 and abs(b.y - a.y) > 0.5:
            lines.append((a.x, a.y, b.x, b.y))
    return lines


def _key(staff: _Staff, bravura, left: float) -> int | None:
    """调号：谱号（行首）或左边小节线后面紧挨着的一串升降号，升为正、降为负；前面的还原号是取消旧调号。

    行首有谱号却没有升降号是 C 大调（0）；小节中间没印调号是 None（沿用前面的）。
    紧挨着符头、同一高度的那个是这个音的临时记号，不算。"""
    clef = next((g for g in bravura if g.code in _CLEFS), None)
    end = clef.x1 if clef is not None else left
    marks = sorted(
        (g for g in bravura if g.code in (_FLAT, _NATURAL, _SHARP) and staff.top - 8 <= g.oy <= staff.bottom + 8 and g.x0 >= end - 1),
        key=lambda g: g.x0,
    )
    chain: list[_Glyph] = []
    for g in marks:
        if g.x0 - (chain[-1].x1 if chain else end) > (1.5 if chain else 6):
            break
        chain.append(g)
    heads = [g for g in bravura if g.code in _HEADS]
    if chain and any(0 <= h.x0 - chain[-1].x1 < 3 and abs(h.oy - chain[-1].oy) < 1 for h in heads):
        chain.pop()
    signs = [g for g in chain if g.code != _NATURAL]
    if not signs:
        return 0 if clef is not None or chain else None
    return len(signs) if signs[0].code == _SHARP else -len(signs)


def _tuning(glyphs, strings: int) -> tuple[int, ...] | None:
    """谱头的调弦：圈起来的弦号后面写“= 音名”（可带降、升号），没写的弦是标准音。"""
    from video2score.tabscore import TUNINGS

    standard = TUNINGS.get(strings)
    if standard is None:
        return None
    tuning = list(standard)
    found = False
    for g in glyphs:
        string = g.code - _STRING_1 + 1
        if "Bravura" not in g.font or not 1 <= string <= strings:
            continue
        letters = [o for o in glyphs if o.text in _PITCH_CLASS and abs(o.oy - g.oy) < 3 and g.x1 < o.x0 < g.x1 + 20]
        if not letters:
            continue
        letter = min(letters, key=lambda o: o.x0)
        pitch = _PITCH_CLASS[letter.text]
        for o in glyphs:
            if o.code in (_FLAT, _SHARP) and abs(o.oy - letter.oy) < 3 and -0.5 < o.x0 - letter.x1 < 3:
                pitch += 1 if o.code == _SHARP else -1
        # 离标准音最近的那个八度（调弦一般只差几个半音）。
        base = standard[string - 1]
        tuning[string - 1] = base + (pitch - base + 6) % 12 - 6
        found = True
    return tuple(tuning) if found else None


def _system_marks(system: _System, row: list[tuple[float, RefMeasure]], glyphs, slides, curves, carry: list[RefEvent], ceiling: float) -> None:
    """一行谱上的技巧和速度。滑音可以跨小节线，所以按整行读。"""
    staff, tab = system.staff, system.tab
    events = [event for _, measure in row for event in measure.events]
    sounding = [event for event in events if not event.rest]
    if carry and sounding:
        sounding[0].marks.add("slide-stop")  # 上一行末尾滑过来的
    carry.clear()
    near = [g for g in glyphs if system.x0 - 2 <= g.cx <= system.x1 + 2 and system.top - 30 <= g.oy <= system.bottom + 14]

    def nearest(x: float, pool: list[RefEvent], limit: float = 8) -> RefEvent | None:
        best = min(pool, key=lambda event: abs(event.x - x), default=None)
        return best if best is not None and abs(best.x - x) <= limit else None

    def before(x: float) -> RefEvent | None:
        return next((event for event in reversed(sounding) if event.x < x), None)

    plain = [g for g in near if g.font.startswith("TimesNewRoman") and "Bold" not in g.font and "Italic" not in g.font]
    if staff is not None and tab is not None:
        # 击弦 H、勾弦 P：五线谱上方和 TAB 上方各印一个，在两个音中间，记在前一个音上。
        # 拍弦 S、勾弦（pop）P：只印在 TAB 上方，在音的正上方。
        over_staff = [g for g in plain if g.text in "HP" and staff.top - 30 <= g.y1 <= staff.top + 1]
        for g in plain:
            if g.text not in "HPS" or not staff.bottom < (g.y0 + g.y1) / 2 < tab.top + 1:
                continue
            legato = g.text != "S" and any(o.text == g.text and abs(o.cx - g.cx) < 1.5 for o in over_staff)
            following = next((event for event in sounding if event.x > g.cx), None)
            if legato and following is not None and following.grace_frets and following.grace_x < g.cx:
                following.grace_marks.add("hammer" if g.text == "H" else "pull")  # 装饰音击（勾）到主音
                continue
            owner = before(g.cx) if legato else nearest(g.cx, sounding)
            if owner is not None:
                owner.marks.add(("hammer" if g.text == "H" else "pull") if legato else ("slap" if g.text == "S" else "pop"))
    bravura = [g for g in near if "Bravura" in g.font]
    for g in bravura:
        if g.code in _ARTICULATIONS:
            owner = nearest(g.cx, events)
            if owner is not None:
                owner.marks.add(_ARTICULATIONS[g.code])
    # 揉弦：一串波浪字形，从揉弦那个音的上方起头。
    waves = sorted((g for g in bravura if g.code == _VIBRATO), key=lambda g: (round(g.oy), g.x0))
    for index, g in enumerate(waves):
        prior = waves[index - 1] if index else None
        if prior is not None and abs(prior.oy - g.oy) < 1 and g.x0 - prior.x1 < 1.5:
            continue
        owner = nearest(g.x0, [event for event in sounding if event.x <= g.x0 + 4], limit=12)
        if owner is not None:
            owner.marks.add("vibrato")
    # 推弦：曲线从推弦的音起头、在 TAB 上方的箭头结束，箭头上写幅度（¼、½、full、1½……以全音计，MusicXML 以半音计，乘 2）。
    for g in bravura:
        if g.code != _BEND_ARROW:
            continue
        text = "".join(o.text for o in sorted(plain, key=lambda o: o.x0) if abs(o.cx - g.cx) < 8 and -2 <= g.y0 - o.y1 < 8)
        whole = "".join(char for char in text if char.isdigit())
        amount = 1 if "full" in text else (int(whole) if whole else 0) + sum(_FRACTIONS.get(char, 0) for char in text)
        start = next((a for a, b in curves if abs(b.x - g.cx) < 2 and g.y0 - 1 <= b.y <= g.y1 + 1), None)
        owner = before(start.x + 1) if start is not None else nearest(g.cx, sounding)
        if owner is not None and amount:
            from video2score.refxml import bend_amount

            owner.marks.add(f"bend-{bend_amount(str(amount * 2))}")
    # 速度：谱表上方（上一行谱下面）的四分音符加“= 数字”，记在它所在的小节。
    for g in glyphs:
        if g.code != _TEMPO_NOTE or not (ceiling + 12 < g.oy < system.top and system.x0 - 20 <= g.cx <= system.x1):
            continue
        digits = sorted((o for o in glyphs if o.font.startswith("TimesNewRoman") and abs(o.oy - g.oy) < 1 and g.x1 < o.x0 < g.x1 + 40 and o.text.isdigit()), key=lambda o: o.x0)
        if digits:
            owner = next((measure for left, measure in reversed(row) if left <= g.cx), row[0][1])
            owner.tempo = float("".join(o.text for o in digits))
    # 滑音：TAB 上的斜线。长的连着前后两个音（跨行的记下来，下一行第一个音收尾）。
    # 短的看线和品位数字的间距：两头都贴着同一根弦上的数字（约 0.7 点）是两音之间的滑音；
    # 起点离前一个数字约 2 点的 45° 短线是滑出（/ doit、\ falloff）；其余是后一个音滑入（/ scoop、\ plop）。
    if tab is None:
        return
    frets = [g for g in near if "Bravura" not in g.font and tab.top - 4 <= (g.y0 + g.y1) / 2 <= tab.bottom + 4 and g.text in "0123456789()X"]
    for x0, y0, x1, y1 in slides:
        if not (system.x0 - 2 <= x0 and x1 <= system.x1 + 2 and tab.top - 4 <= min(y0, y1) and max(y0, y1) <= tab.bottom + 4):
            continue
        rising = y1 < y0
        left = before(x0 + 1)
        right = next((event for event in sounding if event.x > x1 - 1), None)
        if x1 - x0 > 6:
            if left is not None:
                left.marks.add("slide-start")
                if right is not None:
                    right.marks.add("slide-stop")
                else:
                    carry.append(left)
            continue
        # 斜线中点落在哪根弦上，就是哪根弦的滑音。
        middle = (y0 + y1) / 2
        string = 1 + min(range(len(tab.lines)), key=lambda index: abs(tab.lines[index] - middle))
        on_left = left is not None and any(fret.string == string for fret in left.frets)
        on_right = right is not None and any(fret.string == string for fret in right.frets)
        gap_left = min((x0 - g.x1 for g in frets if g.x1 <= x0 + 1 and g.y0 - 1 <= y0 <= g.y1 + 1), default=float("inf"))
        gap_right = min((g.x0 - x1 for g in frets if g.x0 >= x1 - 1 and g.y0 - 1 <= y1 <= g.y1 + 1), default=float("inf"))
        if on_left and on_right and gap_left < 1.5 and gap_right < 1.5:
            left.marks.add("slide-start")
            right.marks.add("slide-stop")
        elif on_left and (gap_left < 3 or not on_right):
            left.marks.add("doit" if rising else "falloff")
        elif right is not None:
            right.marks.add("scoop" if rising else "plop")
