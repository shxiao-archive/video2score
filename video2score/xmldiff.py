"""直接对照两个 MusicXML 的原文：逐小节、逐谱表（五线谱 / TAB）、逐个音比写出来的每一项。

refxml + compare_xml 先把两边都读成"时值、弦品、技巧"再比，双谱表、符杠、符干、闷音、拍号的写法这些
对 Guitar Pro 显示有影响的差别都被抹平了。这里不做解释，按元素比：

- 每小节：拍号（沿用前面的）、调号、每行谱的时值总和和拍号对不对得上；
- 每个音（按起拍位置配对，同一时刻的和弦音按弦或音高配）：有没有、音高、时值、类型、附点、连音、
  延音、符头、符干、符杠、装饰音、弦、品、击勾弦、滑音、连线、演奏法记号、静音（mute / dynamics）、
  推弦幅度、揉弦、延长记号、拍弦勾弦（GP 写在处理指令里）。
  连着的休止先合成一段（拆成几个还是一个都对），只比多长。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from video2score.refxml import bend_amount, gp_attack, parse_musicxml, played_measures

_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# 每个音比哪几项；五线谱和 TAB 各比各的。
_NOTATION_KEYS = ("rest", "pitch", "duration", "type", "dots", "tuplet", "tie", "notehead", "stem", "beam", "slide", "slur", "articulations", "mute", "bend", "vibrato", "fermata", "attack")
_TAB_KEYS = ("rest", "pitch", "duration", "type", "dots", "tuplet", "tie", "notehead", "string", "fret", "technical", "slide", "slur", "articulations", "mute", "silent", "bend", "vibrato", "fermata", "attack")
_NAMES = {
    "missing": "答案有、我们没有",
    "extra": "我们有、答案没有",
    "rest": "休止",
    "pitch": "音高",
    "duration": "时值",
    "type": "音符类型",
    "dots": "附点",
    "tuplet": "连音",
    "tie": "延音",
    "notehead": "符头（闷音 x）",
    "stem": "符干方向",
    "beam": "符杠",
    "string": "弦",
    "fret": "品",
    "technical": "击勾弦",
    "slide": "滑音（slide）",
    "slur": "连线（slur）",
    "articulations": "滑入滑出、跳音",
    "mute": "闷音的 play/mute",
    "silent": "TAB 静音（dynamics=0）",
    "bend": "推弦",
    "vibrato": "揉弦（wavy-line）",
    "fermata": "延长记号",
    "attack": "拍弦勾弦",
}


@dataclass
class _Note:
    onset: Fraction
    grace: int
    features: dict[str, object]


@dataclass
class _Measure:
    number: str
    time: tuple[int, int] | None = None
    key: int | None = None
    staves: dict[str, list[_Note]] = field(default_factory=dict)
    # 每行谱的时值总和（四分音符为 1/4 的分数，和拍号比）。
    totals: dict[str, Fraction] = field(default_factory=dict)


@dataclass
class XmlDiff:
    measures: int = 0
    # 每行谱、每一项：比了几个、不一样几个。
    counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    # 时值总和和拍号对不上的小节（答案、我们各自）。
    overfull: dict[str, list[str]] = field(default_factory=dict)
    meters: list[int] = field(default_factory=lambda: [0, 0])
    keys: list[int] = field(default_factory=lambda: [0, 0])
    lines: list[str] = field(default_factory=list)
    roles: dict[str, tuple[bool, bool]] = field(default_factory=dict)
    # 有差别的小节号（逐个列出，不只报总数），以及答案有、我们整小节没有的。
    differing: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    # TAB 调弦的写法（音名、升降、八度，最低的弦在前）：(我们, 答案)。
    tuning: tuple[list[str], list[str]] = field(default_factory=lambda: ([], []))

    def summary(self) -> str:
        out = [f"直接对照 MusicXML：{self.measures} 小节；拍号 {self.meters[0]}/{self.meters[1]} 小节一样，调号 {self.keys[0]}/{self.keys[1]}"]
        ours, answer = self.tuning
        if ours != answer:
            out.append(f"  调弦写法：答案 {' '.join(answer) or '无'}，我们 {' '.join(ours) or '无'}")
        for role, (ours, answer) in self.roles.items():
            label = "五线谱" if role == "notation" else "TAB"
            if not ours:
                out.append(f"  {label}：我们没有写这一行（答案有）")
                continue
            if not answer:
                out.append(f"  {label}：答案没有这一行")
                continue
            parts = []
            for key, (compared, wrong) in self.counts.get(role, {}).items():
                if wrong:
                    parts.append(f"{_NAMES.get(key, key)} {wrong}/{compared}")
            out.append(f"  {label}：" + ("全一样" if not parts else "；".join(parts)))
        for who, numbers in self.overfull.items():
            if numbers:
                out.append(f"  {who}时值和拍号对不上的小节 {len(numbers)} 个：{', '.join(numbers)}")
        out.append(f"  有差别的小节 {len(self.differing)} 个：" + (", ".join(self.differing) or "无"))
        if self.absent:
            out.append(f"  我们整小节没有的 {len(self.absent)} 个：" + ", ".join(self.absent))
        return "\n".join(out)


def diff_musicxml(ours_path: Path, answer_path: Path) -> XmlDiff:
    ours, answer = _read(ours_path), _read(answer_path)
    result = XmlDiff()
    result.tuning = (_tuning(ours_path), _tuning(answer_path))
    result.roles = {
        role: (any(role in m.staves for m in ours.values()), any(role in m.staves for m in answer.values()))
        for role in ("notation", "tab")
    }
    for who, measures in (("我们", ours), ("答案", answer)):
        result.overfull[who] = [
            number
            for number, measure in measures.items()
            if measure.time and any(total != Fraction(measure.time[0], measure.time[1]) for total in measure.totals.values() if total)
        ]
    for number, want in answer.items():
        got = ours.get(number)
        result.measures += 1
        if got is None:
            result.lines.append(f"第 {number} 小节：我们没有")
            result.absent.append(number)
            continue
        before = len(result.lines)
        result.meters[1] += 1
        result.meters[0] += got.time == want.time
        result.keys[1] += 1
        result.keys[0] += got.key == want.key
        if got.time != want.time:
            result.lines.append(f"第 {number} 小节 拍号：答案 {want.time}，我们 {got.time}")
        for role in ("notation", "tab"):
            if role not in want.staves or role not in got.staves:
                continue
            keys = _NOTATION_KEYS if role == "notation" else _TAB_KEYS
            counts = result.counts.setdefault(role, {})
            for line in _diff_staff(got.staves[role], want.staves[role], keys, counts):
                result.lines.append(f"第 {number} 小节 {'五线谱' if role == 'notation' else 'TAB'} {line}")
        if len(result.lines) > before:
            result.differing.append(number)
    return result


def _tuning(path: Path) -> list[str]:
    """第一处 staff-tuning 的写法，按 line 从低到高："Eb1" 这样（不用 ♭ 字符，GBK 控制台打不出）。"""
    names = []
    for details in parse_musicxml(path).iter("staff-details"):
        lines = sorted(details.findall("staff-tuning"), key=lambda item: int(item.get("line", "0")))
        for tuning in lines:
            alter = int(float(tuning.findtext("tuning-alter") or 0))
            sign = "#" * alter if alter > 0 else "b" * -alter
            names.append(f"{tuning.findtext('tuning-step')}{sign}{tuning.findtext('tuning-octave')}")
        if names:
            break
    return names


def _merge_rests(notes: list[_Note]) -> list[_Note]:
    """连着的休止合成一段：同一段空拍，拆成几个休止（5/4 写成全休止加四分休止）还是一个，都是对的。"""
    merged: list[_Note] = []
    for note in sorted(notes, key=lambda item: (item.onset, item.grace)):
        last = merged[-1] if merged else None
        if (
            note.features.get("rest")
            and last is not None
            and last.features.get("rest")
            and last.onset + last.features["duration"] == note.onset
        ):
            features = dict(last.features)
            features["duration"] = last.features["duration"] + note.features["duration"]
            merged[-1] = _Note(last.onset, last.grace, features)
            continue
        merged.append(note)
    for note in merged:
        if note.features.get("rest"):
            # 休止只比有没有、多长：写成什么类型、带不带附点不算。
            note.features = {**note.features, "type": None, "dots": 0}
    return merged


def _diff_staff(ours: list[_Note], answer: list[_Note], keys: tuple[str, ...], counts: dict[str, list[int]]) -> list[str]:
    lines: list[str] = []
    ours, answer = _merge_rests(ours), _merge_rests(answer)
    remaining = list(ours)
    for key in ("missing", "extra"):
        counts.setdefault(key, [0, 0])
    for want in answer:
        same_time = [note for note in remaining if note.onset == want.onset and note.grace == want.grace]
        got = min(same_time, key=lambda note: _distance(note, want), default=None)
        counts["missing"][0] += 1
        if got is None:
            counts["missing"][1] += 1
            lines.append(f"@{want.onset} 答案有、我们没有：{_short(want)}")
            continue
        remaining.remove(got)
        wrong = []
        for key in keys:
            counts.setdefault(key, [0, 0])
            counts[key][0] += 1
            if got.features.get(key) != want.features.get(key):
                counts[key][1] += 1
                wrong.append(f"{_NAMES.get(key, key)} 答案 {want.features.get(key)!r} 我们 {got.features.get(key)!r}")
        if wrong:
            lines.append(f"@{want.onset}：" + "；".join(wrong))
    for got in remaining:
        counts["extra"][1] += 1
        lines.append(f"@{got.onset} 我们多出：{_short(got)}")
    counts["extra"][0] += len(ours)
    return lines


def _distance(a: _Note, b: _Note) -> int:
    keys = ("string", "pitch", "rest")
    return sum(a.features.get(key) != b.features.get(key) for key in keys)


def _short(note: _Note) -> str:
    f = note.features
    if f.get("rest"):
        return f"休止 {f.get('duration')}"
    text = f"{f.get('type')} {f.get('pitch')}"
    if f.get("string") is not None:
        text += f" {f.get('string')}弦{f.get('fret')}品"
    return text + (" 装饰音" if note.grace else "")


def _read(path: Path) -> dict[str, _Measure]:
    root = parse_musicxml(path)
    part = root.find("part")
    measures: dict[str, _Measure] = {}
    divisions = 1
    time: tuple[int, int] | None = None
    key: int | None = None
    roles: dict[str, str] = {}
    if part is None:
        return measures
    for number, element in played_measures(part):
        number = number or ""
        measure = _Measure(number)
        position = Fraction(0)
        last_onset = Fraction(0)
        graces: Counter = Counter()
        ends: dict[str, Fraction] = {}
        for child in element:
            if child.tag == "attributes":
                divisions = int(child.findtext("divisions") or divisions)
                found = child.find("time")
                if found is not None and found.findtext("beats"):
                    time = (int(found.findtext("beats")), int(found.findtext("beat-type")))
                fifths = child.find("key/fifths")
                if fifths is not None:
                    key = int(fifths.text)
                for clef in child.findall("clef"):
                    roles[clef.get("number", "1")] = "tab" if clef.findtext("sign") == "TAB" else "notation"
            elif child.tag == "backup":
                position -= Fraction(int(child.findtext("duration") or 0), divisions * 4)
            elif child.tag == "forward":
                position += Fraction(int(child.findtext("duration") or 0), divisions * 4)
            elif child.tag == "note":
                staff = child.findtext("staff") or "1"
                role = roles.get(staff, "notation")
                grace = child.find("grace") is not None
                chord = child.find("chord") is not None
                duration = Fraction(int(child.findtext("duration") or 0), divisions * 4)
                onset = last_onset if chord else position
                order = 0
                if grace:
                    graces[(role, onset)] += 0 if chord else 1
                    order = graces[(role, onset)]
                note = _Note(onset, order, _features(child, divisions))
                measure.staves.setdefault(role, []).append(note)
                if not grace and not chord:
                    last_onset = position
                    position += duration
                    ends[role] = max(ends.get(role, Fraction(0)), position)
                elif grace and not chord:
                    last_onset = position
        measure.time, measure.key = time, key
        measure.totals = ends
        measures[number] = measure
    return measures


def _features(note: ET.Element, divisions: int) -> dict[str, object]:
    pitch = note.find("pitch")
    midi = None
    if pitch is not None:
        midi = 12 * (int(pitch.findtext("octave")) + 1) + _STEPS[pitch.findtext("step")] + int(float(pitch.findtext("alter") or 0))
    notations = note.find("notations")
    technical = notations.find("technical") if notations is not None else None
    modification = note.find("time-modification")

    def types(path: str) -> tuple[str, ...]:
        if notations is None:
            return ()
        return tuple(sorted(item.get("type", "") for item in notations.findall(path)))

    articulations = notations.find("articulations") if notations is not None else None
    return {
        "rest": note.find("rest") is not None,
        "pitch": midi,
        "duration": Fraction(int(note.findtext("duration") or 0), divisions * 4),
        "type": note.findtext("type"),
        "dots": len(note.findall("dot")),
        "tuplet": (modification.findtext("actual-notes"), modification.findtext("normal-notes")) if modification is not None else None,
        "tie": tuple(sorted(item.get("type", "") for item in note.findall("tie"))),
        # GP 自己导出时每个音都写 <notehead>normal</notehead>，和不写一样。
        "notehead": None if note.findtext("notehead") == "normal" else note.findtext("notehead"),
        "stem": note.findtext("stem"),
        "beam": tuple((item.get("number", "1"), item.text) for item in note.findall("beam")),
        "string": int(technical.findtext("string")) if technical is not None and technical.findtext("string") else None,
        "fret": int(technical.findtext("fret")) if technical is not None and technical.findtext("fret") else None,
        "technical": tuple(sorted(f"{item.tag}:{item.get('type')}" for item in technical if item.tag in ("hammer-on", "pull-off"))) if technical is not None else (),
        "slide": types("slide"),
        "slur": types("slur"),
        "articulations": tuple(sorted(item.tag for item in articulations if isinstance(item.tag, str))) if articulations is not None else (),
        "mute": note.findtext("play/mute"),
        "silent": note.get("dynamics") == "0",
        "bend": tuple(bend_amount(item.findtext("bend-alter")) for item in technical.findall("bend")) if technical is not None else (),
        "vibrato": types("ornaments/wavy-line"),
        "fermata": notations is not None and notations.find("fermata") is not None,
        "attack": gp_attack(notations),
    }


def run_diff(ours: Path, answer: Path, out: Path | None = None) -> XmlDiff:
    result = diff_musicxml(ours, answer)
    if out is not None:
        out.write_text(result.summary() + "\n\n" + "\n".join(result.lines) + "\n", encoding="utf-8")
    return result
