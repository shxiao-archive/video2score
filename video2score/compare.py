"""把识别结果和标准答案（Guitar Pro 导出的 PDF）按小节号逐小节比对。

python -m video2score compare --job <id> --reference <pdf>，结果写到作业目录的 compare.txt。

比三样：
- 品位：每小节按时间顺序排开的 "弦:品"，算最长公共子序列；
- 节奏：每小节的时值序列（休止记 r）是否完全一样；
- 音：时值和品位都对上的音。
延音的括号不算差别；我们补的休止（时值不足时补在小节末尾）照样参与比对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path

from video2score.reference import RefMeasure, read_reference
from video2score.tabscore import ScoreMeasure, ScoreNote, read_job


@dataclass
class Comparison:
    measures: int = 0
    matched: int = 0
    missing: list[int] = field(default_factory=list)
    extra: list[int] = field(default_factory=list)
    frets: int = 0
    frets_right: int = 0
    notes: int = 0
    notes_right: int = 0
    rhythm_exact: int = 0
    all_exact: int = 0
    lines: list[str] = field(default_factory=list)

    def summary(self) -> str:
        def rate(a: int, b: int) -> str:
            return f"{a}/{b}（{100 * a / b:.1f}%）" if b else "0/0"

        return "\n".join(
            [
                f"标准答案 {self.measures} 小节（不含合并休止），对上号的 {rate(self.matched, self.measures)}",
                f"品位 {rate(self.frets_right, self.frets)}",
                f"音（时值 + 品位都对）{rate(self.notes_right, self.notes)}",
                f"节奏整小节全对 {rate(self.rhythm_exact, self.matched)}，品位和节奏都全对 {rate(self.all_exact, self.matched)}",
                f"识别结果里缺的小节号：{_ranges(self.missing) or '无'}",
                f"识别结果里多出的小节号：{_ranges(self.extra) or '无'}",
            ]
        )


def compare(ours: list[ScoreMeasure], reference: list[RefMeasure]) -> Comparison:
    result = Comparison()
    mine: dict[int, ScoreMeasure] = {}
    for measure in _numbered(ours):
        mine.setdefault(measure.number, measure)
        # 合并休止格代表的那些小节：每个都是一整小节休止（答案是 MusicXML 时它们逐小节写着）。
        for k in range(1, measure.rests_after + 1):
            mine.setdefault(measure.number + k, ScoreMeasure(measure.number + k, [ScoreNote(measure.meter, rest=True)], measure.meter, measure.strings))
    wanted = {measure.number for measure in reference if measure.count == 1}
    multirest = {n for measure in reference if measure.count > 1 for n in range(measure.number, measure.number + measure.count)}
    result.extra = sorted(n for n in mine if n not in wanted and n not in multirest)
    for ref in reference:
        if ref.count > 1:
            continue
        result.measures += 1
        ref_events = [_event_token(e.duration, e.rest, [(f.string, f.fret) for f in e.frets]) for e in ref.events]
        ref_frets = [f"{f.string}:{f.fret if f.fret is not None else 'x'}" for e in ref.events for f in e.frets]
        ref_notes = [token for token, e in zip(ref_events, ref.events) if not e.rest]
        result.frets += len(ref_frets)
        result.notes += len(ref_notes)
        measure = mine.get(ref.number)
        if measure is None:
            result.missing.append(ref.number)
            continue
        result.matched += 1
        our_events = [
            _event_token(n.duration, n.rest or not n.frets, [(f.string, None if f.dead else f.fret) for f in n.frets])
            for n in measure.notes
        ]
        our_frets = [
            f"{f.string}:{'x' if f.dead else f.fret}"
            for n in measure.notes
            if not n.rest
            for f in sorted(n.frets, key=lambda f: f.string)
        ]
        our_notes = [token for token, n in zip(our_events, measure.notes) if not (n.rest or not n.frets)]
        fret_hits = _lcs(ref_frets, our_frets)
        note_hits = _lcs(ref_notes, our_notes)
        result.frets_right += fret_hits
        result.notes_right += note_hits
        rhythm_same = [t.split("|")[0] for t in ref_events] == [t.split("|")[0] for t in our_events]
        result.rhythm_exact += rhythm_same
        exact = ref_events == our_events
        result.all_exact += exact
        if not exact:
            result.lines.append(
                f"第 {ref.number} 小节（PDF 第 {ref.page} 页）"
                + ("" if rhythm_same else " 节奏不同")
                + ("" if fret_hits == len(ref_frets) == len(our_frets) else " 品位不同")
                + f"\n  答案：{' '.join(ref_events)}\n  识别：{' '.join(our_events)}"
                + ("".join(f"\n  备注：{issue}" for issue in measure.issues))
            )
    return result


def _numbered(measures: list[ScoreMeasure]) -> list[ScoreMeasure]:
    """没读出号的小节按前一小节的号接上（前一小节是合并休止时跳过它代表的那些）；
    开头几格读不出号时，按后面第一个读到的号往回推。"""
    first = next((index for index, measure in enumerate(measures) if measure.number is not None), None)
    if first is not None:
        for index in range(first - 1, -1, -1):
            measures[index].number = max(1, measures[index + 1].number - 1 - measures[index].rests_after)
    last: int | None = None
    for measure in measures:
        if measure.number is None and last is not None:
            measure.number = last + 1
        if measure.number is not None:
            last = measure.number + measure.rests_after
    return [measure for measure in measures if measure.number is not None]


def _event_token(duration, rest: bool, frets: list[tuple[int, int | None]]) -> str:
    text = ("r" if rest else "") + str(duration)
    if rest:
        return text
    return text + "|" + "+".join(f"{s}:{f if f is not None else 'x'}" for s, f in sorted(frets))


def _lcs(a: list[str], b: list[str]) -> int:
    return sum(block.size for block in SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks())


def _ranges(numbers: list[int]) -> str:
    parts: list[str] = []
    for n in sorted(numbers):
        if parts and "-" not in parts[-1].split(",")[-1] and int(parts[-1]) == n - 1:
            parts[-1] = f"{parts[-1]}-{n}"
        elif parts and "-" in parts[-1] and int(parts[-1].split("-")[1]) == n - 1:
            parts[-1] = f"{parts[-1].split('-')[0]}-{n}"
        else:
            parts.append(str(n))
    return ",".join(parts)


def run_compare(job_dir: Path, pdf: Path) -> Comparison:
    result = compare(read_job(job_dir), read_reference(pdf))
    text = result.summary() + "\n\n" + "\n\n".join(result.lines) + "\n"
    (job_dir / "compare.txt").write_text(text, encoding="utf-8")
    return result


@dataclass
class Structure:
    """按内容对齐后的结构比对：丢了哪些小节、多出哪些、多出的里面有几个是重复。"""

    expected: int = 0
    ours: int = 0
    missing: list[int] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    duplicates: int = 0
    # 结尾多出来、和最后一个对上的小节一模一样的几格（视频把最后的延音小节多放了几遍），和原谱等效。
    tail_repeats: int = 0

    def summary(self) -> str:
        return (
            f"原谱 {self.expected} 小节，拼出 {self.ours} 小节；"
            f"丢了 {len(self.missing)} 个：{_ranges(self.missing) or '无'}；"
            f"多出 {len(self.extra)} 个（其中 {self.duplicates} 个是和邻格一样的重复）"
            + (f"：{'，'.join(self.extra[:12])}" if self.extra else "")
            + (f"；结尾另有 {self.tail_repeats} 格重复最后一小节（等效）" if self.tail_repeats else "")
        )


def align_structure(ours: list[ScoreMeasure], reference: list[RefMeasure]) -> Structure:
    """把原谱的小节序列和我们拼出的小节序列按内容对齐（允许缺、多）。

    小节号可能读错，所以主要看内容：每小节按时间顺序的"弦:品"序列算相似度，号一样再加一点。
    合并休止按它代表的小节数展开成几个空小节。动态规划找得分最高的对齐。
    """
    ref_units: list[tuple[int, list[str]]] = []
    for measure in reference:
        if measure.count > 1:
            ref_units.extend((measure.number + k, []) for k in range(measure.count))
        else:
            tokens = [f"{f.string}:{f.fret if f.fret is not None else 'x'}" for e in measure.events for f in e.frets]
            ref_units.append((measure.number, tokens))
    our_units: list[tuple[int | None, list[str]]] = []
    for measure in ours:
        tokens = [
            f"{f.string}:{'x' if f.dead else f.fret}"
            for n in measure.notes
            if not n.rest
            for f in sorted(n.frets, key=lambda f: f.string)
        ]
        our_units.append((measure.number, tokens))
        our_units.extend(
            ((measure.number + k + 1) if measure.number is not None else None, []) for k in range(measure.rests_after)
        )

    def similarity(a: tuple[int | None, list[str]], b: tuple[int | None, list[str]]) -> float:
        if not a[1] and not b[1]:
            value = 0.8
        elif not a[1] or not b[1]:
            value = 0.0
        else:
            value = 2.0 * _lcs(a[1], b[1]) / (len(a[1]) + len(b[1]))
        if a[0] is not None and a[0] == b[0]:
            value += 0.3
        return value

    gap = 0.35
    n, m = len(ref_units), len(our_units)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0], move[i][0] = -gap * i, "up"
    for j in range(1, m + 1):
        score[0][j], move[0][j] = -gap * j, "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                # 完全不像的一对（0 分）配起来不如记成一缺一多（-0.7）。
                (score[i - 1][j - 1] + similarity(ref_units[i - 1], our_units[j - 1]) - 0.8, "diag"),
                (score[i - 1][j] - gap, "up"),
                (score[i][j - 1] - gap, "left"),
            )
            score[i][j], move[i][j] = max(options)
    result = Structure(expected=n, ours=m)
    i, j = n, m
    extra_units: list[int] = []
    paired: dict[int, int] = {}
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "diag":
            paired[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif step == "up":
            result.missing.append(ref_units[i - 1][0])
            i -= 1
        else:
            extra_units.append(j - 1)
            j -= 1
    result.missing.sort()
    # 结尾连着多出来的格子，和最后一个对上的小节内容一样：视频多放了几遍最后的延音，不算多。
    trailing = sorted(extra_units)
    matched_last = max(paired, default=None)
    while (
        trailing
        and trailing[-1] == m - 1 - result.tail_repeats
        and matched_last is not None
        and paired[matched_last] == n - 1
        and trailing[-1] > matched_last
        and our_units[trailing[-1]][1] == our_units[matched_last][1]
    ):
        trailing.pop()
        result.tail_repeats += 1
    for index in trailing:
        unit = our_units[index]
        neighbours = [our_units[k] for k in (index - 1, index + 1) if 0 <= k < m]
        duplicate = bool(unit[1]) and any(unit[1] == other[1] for other in neighbours)
        result.duplicates += duplicate
        result.extra.append(f"第 {index + 1} 格（号 {unit[0] if unit[0] is not None else '?'}）" + ("重复" if duplicate else ""))
    return result


# 逐类比对的技巧名 → 报告里的中文名。延音线、滑音只数起点。
_MARK_NAMES = {
    "tie-start": "延音线",
    "slide-start": "滑音（两音之间）",
    "scoop": "滑入",
    "doit": "向上滑出",
    "falloff": "向下滑出",
    "hammer": "击弦",
    "pull": "勾弦",
    "staccato": "跳音",
    "accent": "重音（>）",
    "strong-accent": "重音（^）",
    "grace": "装饰音",
    "dead": "闷音",
    "slap": "拍弦",
    "pop": "勾弦（pop）",
    "bend-0.5": "推弦 0.5",
    "bend-1": "推弦 1",
    "bend-1.5": "推弦 1.5",
    "bend-2": "推弦 2",
    "vibrato": "揉弦",
    "fermata": "延长记号",
}


@dataclass
class XmlComparison:
    """导出的 MusicXML 和答案 MusicXML 逐项比：核心（时值、弦品）、每类技巧、谱面记号。"""

    core: Comparison = field(default_factory=Comparison)
    # 技巧名 → [答案里有、我们也有；答案里有、我们没有；我们有、答案里没有]
    marks: dict[str, list[int]] = field(default_factory=dict)
    tuning: tuple[tuple[int, ...] | None, tuple[int, ...] | None] = (None, None)
    keys: list[int] = field(default_factory=lambda: [0, 0])  # [对, 总]
    meters: list[int] = field(default_factory=lambda: [0, 0])
    tempo: tuple[float | None, float | None] = (None, None)

    def summary(self) -> str:
        lines = [self.core.summary()]
        tuning_ok = "对" if self.tuning[0] == self.tuning[1] else f"不对（我们 {self.tuning[0]}，答案 {self.tuning[1]}）"
        ours, answer = self.tempo
        if ours == answer:
            tempo_ok = "对"
        elif ours is not None and answer is not None and abs(ours - answer) <= 2:
            # 画面里没印速度时按视频时间估，差一两拍算对上。
            tempo_ok = f"约对（我们 {ours:g}，答案 {answer:g}）"
        else:
            tempo_ok = f"不对（我们 {ours}，答案 {answer}）"
        lines.append(f"调弦{tuning_ok}；速度{tempo_ok}；调号 {self.keys[0]}/{self.keys[1]} 小节对；拍号 {self.meters[0]}/{self.meters[1]} 小节对")
        parts = []
        for name, label in _MARK_NAMES.items():
            hit, miss, false = self.marks.get(name, [0, 0, 0])
            if hit or miss or false:
                parts.append(f"{label} {hit}/{hit + miss}" + (f"（多 {false}）" if false else ""))
        lines.append("技巧（找到/答案里有）：" + ("，".join(parts) or "无"))
        return "\n".join(lines)


def compare_xml(ours_path: Path, answer_path: Path) -> XmlComparison:
    from video2score.refxml import read_musicxml, read_tuning

    ours = read_musicxml(ours_path)
    answer = read_musicxml(answer_path)
    result = XmlComparison(core=_compare_ref(ours, answer), tuning=(read_tuning(ours_path), read_tuning(answer_path)))
    mine = {measure.number: measure for measure in ours}
    result.tempo = (next((m.tempo for m in ours if m.tempo), None), next((m.tempo for m in answer if m.tempo), None))
    our_key = answer_key = None
    our_meter = answer_meter = None
    for measure in answer:
        other = mine.get(measure.number)
        answer_key = measure.key if measure.key is not None else answer_key
        answer_meter = measure.meter or answer_meter
        if other is None:
            continue
        our_key = other.key if other.key is not None else our_key
        our_meter = other.meter or our_meter
        result.keys[1] += 1
        result.keys[0] += (our_key or 0) == (answer_key or 0)
        result.meters[1] += 1
        result.meters[0] += our_meter == answer_meter
        wanted, found = _mark_positions(measure), _mark_positions(other)
        for name in _MARK_NAMES:
            a, b = wanted.get(name, []), found.get(name, [])
            hit = sum(min(a.count(x), b.count(x)) for x in set(a))
            counts = result.marks.setdefault(name, [0, 0, 0])
            counts[0] += hit
            counts[1] += len(a) - hit
            counts[2] += len(b) - hit
    for measure in ours:
        if measure.number not in {m.number for m in answer}:
            for name, positions in _mark_positions(measure).items():
                result.marks.setdefault(name, [0, 0, 0])[2] += len(positions)
    return result


def _mark_positions(measure: RefMeasure) -> dict[str, list[Fraction]]:
    """每类技巧出现在小节里的哪些起拍位置（同一位置可以有好几个）。"""
    positions: dict[str, list[Fraction]] = {}
    onset = Fraction(0)
    for event in measure.events:
        for mark in event.marks:
            if mark in _MARK_NAMES:
                positions.setdefault(mark, []).append(onset)
        positions.setdefault("grace", []).extend([onset] * event.graces)
        positions.setdefault("dead", []).extend(onset for fret in event.frets if fret.fret is None and not event.rest)
        onset += event.duration
    return positions


def _compare_ref(ours: list[RefMeasure], reference: list[RefMeasure]) -> Comparison:
    """compare() 的两边都是 RefMeasure 的版本（我们导出的 MusicXML 读回来）。"""
    result = Comparison()
    mine = {measure.number: measure for measure in ours}
    wanted = {measure.number for measure in reference}
    result.extra = sorted(n for n in mine if n not in wanted)
    for ref in reference:
        result.measures += 1
        ref_events = _rest_measure(ref) or [_event_token(e.duration, e.rest, [(f.string, f.fret) for f in e.frets]) for e in ref.events]
        ref_frets = [f"{f.string}:{f.fret if f.fret is not None else 'x'}" for e in ref.events for f in e.frets]
        ref_notes = [token for token, e in zip(ref_events, ref.events) if not e.rest]
        result.frets += len(ref_frets)
        result.notes += len(ref_notes)
        measure = mine.get(ref.number)
        if measure is None:
            result.missing.append(ref.number)
            continue
        result.matched += 1
        our_events = _rest_measure(measure) or [
            _event_token(e.duration, e.rest or not e.frets, [(f.string, f.fret) for f in e.frets]) for e in measure.events
        ]
        our_frets = [f"{f.string}:{f.fret if f.fret is not None else 'x'}" for e in measure.events if not e.rest for f in e.frets]
        our_notes = [token for token, e in zip(our_events, measure.events) if not (e.rest or not e.frets)]
        fret_hits = _lcs(ref_frets, our_frets)
        result.frets_right += fret_hits
        result.notes_right += _lcs(ref_notes, our_notes)
        rhythm_same = [t.split("|")[0] for t in ref_events] == [t.split("|")[0] for t in our_events]
        result.rhythm_exact += rhythm_same
        exact = ref_events == our_events
        result.all_exact += exact
        if not exact:
            result.lines.append(
                f"第 {ref.number} 小节"
                + ("" if rhythm_same else " 节奏不同")
                + ("" if fret_hits == len(ref_frets) == len(our_frets) else " 品位不同")
                + f"\n  答案：{' '.join(ref_events)}\n  识别：{' '.join(our_events)}"
            )
    return result


def _rest_measure(measure: RefMeasure) -> list[str] | None:
    """整小节都是休止：不管拆成几个休止符（5/4 写成全休止加四分休止），都记成一个总时值的休止。"""
    if measure.events and all(event.rest for event in measure.events):
        return [_event_token(sum((event.duration for event in measure.events), Fraction(0)), True, [])]
    return None
