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
    # 演奏技巧（MusicXML 答案才有）：slide-start、slide-stop、scoop、doit、falloff、hammer、pull、staccato、tie-start、tie-stop。
    marks: set[str] = field(default_factory=set)
    # 紧挨在它前面的装饰音个数（装饰音本身不算事件，不占时值）。
    graces: int = 0


@dataclass
class RefMeasure:
    number: int
    page: int
    events: list[RefEvent]
    meter: Fraction | None = None
    # 合并休止：这一格代表几个小节。
    count: int = 1
    # 调号（升号数为正、降号数为负）和速度（每分钟四分音符数），只在印了的小节有（MusicXML 答案才有）。
    key: int | None = None
    tempo: float | None = None

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
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            measures.extend(_read_page(page, page_number, measures))
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
    return measures


def _read_page(page, page_number: int, before: list[RefMeasure]) -> list[RefMeasure]:
    glyphs = _glyphs(page)
    drawings = page.get_drawings()
    systems = _systems(drawings)
    stems, beams, bars = _strokes(drawings)
    out: list[RefMeasure] = []
    for system in systems:
        edges = _measure_edges(system, bars)
        for left, right in zip(edges, edges[1:]):
            if right - left < 8:
                continue
            previous = out[-1] if out else (before[-1] if before else None)
            measure = _read_measure(system, left, right, glyphs, stems, beams, page_number, previous)
            out.append(measure)
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
                stems.append((a.x, min(a.y, b.y), max(a.y, b.y)))
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
    top = system.top - 12
    inside = [g for g in glyphs if left - 0.5 <= g.cx <= right + 0.5 and top <= (g.y0 + g.y1) / 2 <= system.bottom + 12]
    number = _number(glyphs, system, left)
    if number is None:
        number = previous.number + previous.count if previous else 1
    measure = RefMeasure(number, page_number, [])
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
        measure.events = _staff_events(staff, bravura, inside, stems, beams, left, right)
    frets = _frets(tab, inside) if tab is not None else []
    for event in measure.events:
        if event.rest:
            continue
        event.frets = [fret for x, fret in frets if abs(x - event.x) < 3.5]
    if measure.count > 1:
        measure.events = []
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


def _staff_events(staff, bravura, inside, stems, beams, left, right) -> list[RefEvent]:
    heads = [g for g in bravura if g.code in _HEADS]
    full = max(g.size for g in heads) if heads else 0
    # 装饰音的符头小一号，不占时值。
    heads = [g for g in heads if g.size >= 0.9 * full]
    flags = [g for g in bravura if g.code in _FLAGS]
    dots = [g for g in bravura if g.code == _DOT]
    events: list[RefEvent] = []
    used_stems: dict[int, RefEvent] = {}
    for head in sorted(heads, key=lambda g: g.x0):
        y = head.oy
        stem_index = None
        for index, (x, y0, y1) in enumerate(stems):
            if head.x0 - 0.8 <= x <= head.x1 + 0.8 and y0 - 2 <= y <= y1 + 2 and left <= x <= right:
                stem_index = index
                break
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
    return events


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


def _frets(tab: _Staff, inside) -> list[tuple[float, RefFret]]:
    """TAB 上的品位：同一行里挨着的字符拼成一个数，括号是延音，X 是闷音。"""
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
    chars = [g for g in chars if g.size >= 0.9 * full]
    chars.sort(key=lambda g: (round((g.y0 + g.y1) / 2), g.x0))
    tokens: list[list[_Glyph]] = []
    for g in chars:
        if tokens:
            last = tokens[-1][-1]
            if abs((last.y0 + last.y1) / 2 - (g.y0 + g.y1) / 2) < 1 and g.x0 - last.x1 < 1.0:
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
