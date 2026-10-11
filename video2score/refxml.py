"""读 MusicXML 格式的标准答案（五线谱 + TAB 双谱表），转成和 reference.py 一样的 RefMeasure 列表。

五线谱那行给时值、休止、附点、连音；TAB 那行（带 <string>/<fret> 的音）给弦和品，按起拍位置配到五线谱的音上。
只有一行谱时（我们自己导出的纯 TAB），那一行既给时值又给品。
装饰音不算事件（原谱比对也不算），个数记在它后面那个音的 graces 上，它的技巧也并给那个音。技巧（滑音、击勾弦、跳音、延音线、
推弦、揉弦、延长记号、拍弦勾弦）两行谱上的合在一起。拍弦勾弦 GP 写成 <notations> 里的处理指令
<?GP <root><bass-attack type="slap"/></root>?>，所以解析时要保留处理指令。
合并休止在 MusicXML 里是逐小节写的全休止，照样一小节一格。
答案里的反复记号（反复线、几房跳跃）按演奏顺序展开、从头连续编号：视频里的 GP 画面是展开画的（比如 38~41 演两遍，
第二遍画成 42~45），展开后才和视频的小节号对得上。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from video2score.reference import RefEvent, RefFret, RefMeasure

# 这些 MusicXML 元素（出现在音的 notations 里）记成对应的技巧名。
_ARTICULATIONS = {
    "scoop": "scoop",
    "doit": "doit",
    "falloff": "falloff",
    "staccato": "staccato",
    "accent": "accent",
    "strong-accent": "strong-accent",
}
_TECHNICAL = {"hammer-on": "hammer", "pull-off": "pull"}
_GP_ATTACK = re.compile(r'<bass-attack\s+type="(\w+)"')
_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass
class _Note:
    onset: Fraction
    event: RefEvent
    frets: list[RefFret]


def parse_musicxml(path: Path) -> ET.Element:
    """读 MusicXML，保留 <?GP ...?> 处理指令（GP 用它写拍弦、勾弦、揉弦的类型）。"""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_pis=True))
    return ET.parse(path, parser=parser).getroot()


def gp_attack(notations: ET.Element | None) -> str | None:
    """<notations> 里 GP 处理指令写的拍弦（slap）、勾弦（pop）。"""
    if notations is None:
        return None
    for item in notations:
        if item.tag is ET.ProcessingInstruction:
            found = _GP_ATTACK.search(item.text or "")
            if found:
                return found.group(1)
    return None


def bend_amount(text: str | None) -> str:
    """推弦幅度（全音为 1）统一写法：0.5、1、1.5。"""
    value = float(text or 0)
    return str(int(value)) if value == int(value) else str(value)


def read_musicxml(path: Path) -> list[RefMeasure]:
    root = parse_musicxml(path)
    part = root.find("part")
    if part is None:
        return []
    measures: list[RefMeasure] = []
    divisions = 1
    meter: Fraction | None = None
    for number, element in played_measures(part):
        attributes = element.find("attributes")
        key: int | None = None
        if attributes is not None:
            divisions = int(attributes.findtext("divisions", default=str(divisions)))
            time = attributes.find("time")
            if time is not None and time.findtext("beats") and time.findtext("beat-type"):
                meter = Fraction(int(time.findtext("beats")), int(time.findtext("beat-type")))
            if attributes.find("key/fifths") is not None:
                key = int(attributes.findtext("key/fifths"))
        tempo = element.findtext(".//direction/direction-type/metronome/per-minute")
        staves = _staff_notes(element, divisions)
        notation = staves.get("1", [])
        tab = staves.get("2", [])
        if not notation:
            notation, tab = tab, []
        by_onset: dict[Fraction, list[_Note]] = {}
        for note in tab:
            by_onset.setdefault(note.onset, []).append(note)
        events: list[RefEvent] = []
        for note in notation:
            event = note.event
            partners = by_onset.get(note.onset, [])
            if not event.rest:
                frets = [fret for partner in partners for fret in partner.frets]
                event.frets = frets or note.frets
            for partner in partners:
                event.marks |= partner.event.marks
                event.graces = max(event.graces, partner.event.graces)
            events.append(event)
        number = _number(number, measures)
        measures.append(
            RefMeasure(number=number, page=0, events=events, meter=meter, key=key, tempo=float(tempo) if tempo else None)
        )
    return measures


def read_tuning(path: Path) -> tuple[int, ...] | None:
    """TAB 谱表的调弦（MIDI 音高），第 1 弦（音最高）在前。没写就返回 None。"""
    root = ET.parse(path).getroot()
    for details in root.iter("staff-details"):
        lines = []
        for tuning in details.findall("staff-tuning"):
            step = tuning.findtext("tuning-step")
            octave = tuning.findtext("tuning-octave")
            if step is None or octave is None:
                continue
            alter = int(float(tuning.findtext("tuning-alter", default="0")))
            lines.append((int(tuning.get("line", "0")), 12 * (int(octave) + 1) + _STEPS[step] + alter))
        if lines:
            # line 1 是最下面那条线（音最低的弦）。
            return tuple(midi for _line, midi in sorted(lines, reverse=True))
    return None


def played_measures(part: ET.Element) -> list[tuple[str | None, ET.Element]]:
    """按演奏顺序排好的 (小节号, 小节)。没有反复记号时就是原样、原号；有的话展开，从第一小节的号起连续编号。"""
    written = part.findall("measure")
    if not any(barline.find("repeat") is not None for measure in written for barline in measure.findall("barline")):
        return [(measure.get("number"), measure) for measure in written]
    order: list[ET.Element] = []
    start = 0  # 往回跳到哪一小节（最近的正向反复线，没有就是开头）
    passes = 1  # 现在是这一段反复的第几遍
    jumped: dict[int, int] = {}  # 每道反向反复线已经往回跳了几次
    index = 0
    while index < len(written):
        measure = written[index]
        if _repeat(measure, "forward") is not None and passes == 1:
            start = index
        ending = _ending(measure, "start")
        if ending is not None and passes not in ending:
            # 不是这一遍的房子：跳到这个房子结束的那一小节后面。
            index = _ending_end(written, index) + 1
            continue
        order.append(measure)
        backward = _repeat(measure, "backward")
        if backward is not None:
            times = int(backward.get("times") or 2)
            if jumped.get(index, 0) < times - 1:
                jumped[index] = jumped.get(index, 0) + 1
                passes += 1
                index = start
                continue
            passes = 1
            start = index + 1
        index += 1
    first = _number(written[0].get("number"), [])
    return [(str(first + offset), measure) for offset, measure in enumerate(order)]


def _repeat(measure: ET.Element, direction: str) -> ET.Element | None:
    for barline in measure.findall("barline"):
        repeat = barline.find("repeat")
        if repeat is not None and repeat.get("direction") == direction:
            return repeat
    return None


def _ending(measure: ET.Element, kind: str) -> set[int] | None:
    """这一小节上某种房子记号（start、stop、discontinue）管的是第几遍。"""
    for barline in measure.findall("barline"):
        ending = barline.find("ending")
        if ending is not None and ending.get("type") == kind:
            return {int(text) for text in re.findall(r"\d+", ending.get("number") or "")}
    return None


def _ending_end(measures: list[ET.Element], index: int) -> int:
    for later in range(index, len(measures)):
        if _ending(measures[later], "stop") is not None or _ending(measures[later], "discontinue") is not None:
            return later
    return index


def _number(text: str | None, before: list[RefMeasure]) -> int:
    try:
        return int(text or "")
    except ValueError:
        return before[-1].number + 1 if before else 1


def _staff_notes(measure: ET.Element, divisions: int) -> dict[str, list[_Note]]:
    """按谱表分开的音（起拍位置、事件、品位）。和弦的几个音并成一个事件。"""
    whole = Fraction(4 * divisions)
    position = Fraction(0)
    last_start = Fraction(0)
    staves: dict[str, list[_Note]] = {}
    graces: dict[str, int] = {}
    grace_marks: dict[str, set[str]] = {}
    for child in measure:
        if child.tag == "backup":
            position -= Fraction(int(child.findtext("duration", default="0")))
            continue
        if child.tag == "forward":
            position += Fraction(int(child.findtext("duration", default="0")))
            continue
        if child.tag != "note":
            continue
        staff = child.findtext("staff", default="1")
        if child.find("grace") is not None:
            graces[staff] = graces.get(staff, 0) + 1
            # 装饰音上的击勾弦、滑音算在它后面那个音上。
            grace_marks.setdefault(staff, set()).update(mark for mark in _marks(child) if not mark.startswith("tie"))
            continue
        duration = Fraction(int(child.findtext("duration", default="0")))
        chord = child.find("chord") is not None
        start = last_start if chord else position
        frets = _frets(child)
        marks = _marks(child)
        if chord and staves.get(staff):
            previous = staves[staff][-1]
            previous.frets.extend(frets)
            previous.event.marks |= marks
            continue
        modification = child.find("time-modification")
        tuplet = None
        if modification is not None:
            tuplet = Fraction(int(modification.findtext("normal-notes")), int(modification.findtext("actual-notes")))
        event = RefEvent(
            x=float(start),
            duration=duration / whole,
            rest=child.find("rest") is not None,
            dots=len(child.findall("dot")),
            tuplet=tuplet,
            marks=marks | grace_marks.pop(staff, set()),
            graces=graces.pop(staff, 0),
        )
        staves.setdefault(staff, []).append(_Note(start / whole, event, frets))
        last_start = start
        position = start + duration
    return staves


def _marks(note: ET.Element) -> set[str]:
    marks: set[str] = set()
    for tie in note.findall("tie"):
        marks.add(f"tie-{tie.get('type')}")
    for notations in note.findall("notations"):
        for slide in notations.findall("slide") + notations.findall("glissando"):
            marks.add(f"slide-{slide.get('type')}")
        for tag, name in _ARTICULATIONS.items():
            if notations.find(f"articulations/{tag}") is not None:
                marks.add(name)
        for tag, name in _TECHNICAL.items():
            item = notations.find(f"technical/{tag}")
            if item is not None and item.get("type", "start") == "start":
                marks.add(name)
        for bend in notations.findall("technical/bend"):
            marks.add(f"bend-{bend_amount(bend.findtext('bend-alter'))}")
        if any(line.get("type") == "start" for line in notations.findall("ornaments/wavy-line")):
            marks.add("vibrato")
        if notations.find("fermata") is not None:
            marks.add("fermata")
        attack = gp_attack(notations)
        if attack:
            marks.add(attack)
    return marks


def _frets(note: ET.Element) -> list[RefFret]:
    technical = note.find("notations/technical")
    if technical is None or technical.findtext("string") is None:
        return []
    string = int(technical.findtext("string"))
    fret_text = technical.findtext("fret")
    # 闷音：我们导出的写 x 符头；GP 自己导出的符头照常，只写 <play><mute>straight</mute></play>，品位照留。
    dead = note.findtext("notehead") == "x" or note.findtext("play/mute") == "straight"
    tied = any(tie.get("type") == "stop" for tie in note.findall("tie"))
    fret = None if dead or fret_text is None else int(fret_text)
    return [RefFret(string, fret, tied)]
