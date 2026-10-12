"""把小节库里的五线谱节奏和 TAB 品位合在一起，写成带弦、品的 MusicXML。

Guitar Pro 8 和 MuseScore 都能导入 MusicXML，再从里面导出 .gp 和 MIDI。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

from video2score.header import key_signature
from video2score.rhythm import BASE, Event, fermatas, read_staff_rhythm, staff5, stem_event, whole_event
from video2score.tab import Bank, FretNote, Onset, adapted_tab_bank, bass_attacks, bends, fill_frets, mark_arcs, onsets, read_frets, tab_staff, vibratos, vote_frets
from video2score.tabrhythm import multirest_digits, read_multirest, read_tab_rhythm, time_signature
from video2score.tidy import geometry, has_multirest

# 空弦音高（MIDI），第 1 弦（最细、音最高）在前。
TUNINGS = {
    4: (43, 38, 33, 28),  # 贝斯 G D A E
    5: (43, 38, 33, 28, 23),  # 五弦贝斯 G D A E B
    6: (64, 59, 55, 50, 45, 40),  # 吉他 E B G D A E
    7: (64, 59, 55, 50, 45, 40, 35),
}
# 一个四分音符分成 480 份：三连音、五连音、六十四分音符都能用整数表示。
DIVISIONS = 480
_STEPS = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
_ALTERS = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)
_METERS = {Fraction(2, 4), Fraction(3, 4), Fraction(4, 4), Fraction(5, 4), Fraction(6, 4), Fraction(7, 4)}


@dataclass
class ScoreNote:
    duration: Fraction
    rest: bool = False
    frets: list[FretNote] = field(default_factory=list)
    dots: int = 0
    kind: str = "quarter"
    tie_from_previous: bool = False
    tie_to_next: bool = False
    # 连音：实际时值和正常时值之比（三连音是 2/3），以及括号的头尾。
    tuplet: Fraction | None = None
    tuplet_mark: str | None = None
    # 五线谱上符头的高度（从第一线往下数几个线距）。延音前后两个音高度一样。
    staff_y: float | None = None
    # 演奏技巧：slide-start / slide-stop（两音之间的滑音）、scoop（从下滑入）、plop（从上滑入）、
    # doit（向上滑出）、falloff（向下滑出）、hammer、pull、staccato。
    marks: set[str] = field(default_factory=set)
    # 装饰音：不占时值，写在后面那个音前面。
    grace: bool = False
    # 五线谱上的符干方向（"up" / "down"，不知道是 None）和与后一个音之间有没有符杠（不知道是 None）。
    stem: str | None = None
    beam_next: bool | None = None
    # 五线谱上有音、TAB 上没数字、又接不上前一个音的延音：先用休止占位（多认的符头常是这种）。
    placeholder: bool = False
    # 小节图里的横坐标；两音之间那道斜线归谁（"out" 前一个音滑出、"in" 后一个音滑入，按五线谱上斜笔的高度定）。
    x: float | None = None
    slide_owner: str | None = None
    # 和前一个音之间，五线谱上有没有延音弧线（同一小节里才看；看不了是 None）。
    tie_arc: bool | None = None


@dataclass
class ScoreMeasure:
    number: int | None
    notes: list[ScoreNote]
    meter: Fraction
    strings: int
    rests_after: int = 0
    issues: list[str] = field(default_factory=list)
    # 小节开头印着拍号：读出来的拍号；印了但没读出来时 signature 是 None、new_meter 为真。
    signature: Fraction | None = None
    new_meter: bool = False
    # 小节开头印着调号时读出的升降号数（升为正）；没印或没读出是 None，沿用前面的调号。
    fifths: int | None = None
    # 这一格里的推弦箭头：(横坐标, 半音数, TAB 线距)。归哪个音要看前面的小节，由 _link_bends 定。
    bends: list = field(default_factory=list)
    # 这一格第一次出现在画面上的时间（秒）和出现的帧数，估速度用。
    seconds: float | None = None
    seen: int = 0
    # 小节库里这一格的干净图（相对作业目录），界面上列有问题的小节时显示。
    image: str | None = None


@dataclass
class Marks:
    """全曲的谱面记号：调弦（第 1 弦在前的 MIDI 音高）、调号（升号数为正）、速度（每分钟四分音符数）。读不出的是 None。"""

    tuning: tuple[int, ...] | None = None
    key: int | None = None
    tempo: int | None = None
    # 速度是按视频时间估出来的（画面里没印速度）。
    tempo_estimated: bool = False


def score_marks(job_dir: Path) -> Marks:
    """读乐谱开头的调弦字母（measures/prefix.png）、第一小节行首的调号、上方的速度。"""
    from video2score.clean import CleanMeasure
    from video2score.header import key_fifths
    from video2score.marks import read_tempo, read_tuning

    library = json.loads((job_dir / "measures.json").read_text(encoding="utf-8"))
    if not library:
        return Marks()
    first = library[0]
    marks = Marks()
    tab = tab_staff(first)
    prefix_path = job_dir / "measures" / "prefix.png"
    if tab is not None and prefix_path.is_file():
        prefix = cv2.imread(str(prefix_path), cv2.IMREAD_GRAYSCALE)
        standard = TUNINGS.get(len(tab.lines))
        if prefix is not None and standard is not None:
            marks.tuning = read_tuning(prefix, list(tab.lines), standard)
    heads = []
    for record in library[:3]:
        image = cv2.imread(str(job_dir / record["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        heads.append(
            CleanMeasure(
                image=image,
                spacing=float(record["spacing"]),
                staff_top=float(record["staff_top"]),
                lines=tuple(record["lines"]),
                layout=tuple(tuple(item) for item in record["layout"]),
                kind=record["kind"],
                left_bar=int(record["bars"][0]),
                right_bar=int(record["bars"][1]),
            )
        )
    if heads:
        marks.key = key_fifths(heads)
        top = heads[0].staff_top + heads[0].lines[0]
        marks.tempo = read_tempo(heads[0].image, top, heads[0].spacing, (heads[0].left_bar, heads[0].right_bar))
    return marks


def read_job(job_dir: Path) -> list[ScoreMeasure]:
    """读一个作业的小节库，返回每小节的音符。"""
    library = json.loads((job_dir / "measures.json").read_text(encoding="utf-8"))
    loaded = []
    for record in library:
        # 参考帧在前，后面是另外几帧（和参考帧对齐过）；每帧各读一遍再投票。
        paths = [record.get("raw") or record["image"]] + list(record.get("raws") or [])
        images = [cv2.imread(str(job_dir / path), cv2.IMREAD_GRAYSCALE) for path in paths]
        images = [image for image in images if image is not None]
        clean = cv2.imread(str(job_dir / record["image"]), cv2.IMREAD_GRAYSCALE) if record.get("raw") else None
        loaded.append((record, images, tab_staff(record), staff5(record), clean))
    loaded = _cut_merged(loaded)
    library = [record for record, *_rest in loaded]
    starts = _content_starts(loaded)
    thick = _upscaled(job_dir)
    opening = _opening_key(loaded)
    bank = adapted_tab_bank([(images[0], tab, tuple(record["bars"])) for record, images, tab, _staff, _clean in loaded if tab])
    candidates = [
        _read_measure(record, images, tab, staff, bank, clean, start, thick, opening)
        for (record, images, tab, staff, clean), start in zip(loaded, starts)
    ]
    measures = _choose(candidates)
    for measure, record in zip(measures, library):
        measure.seconds, measure.seen = record.get("first_seconds"), int(record.get("seen") or 0)
        measure.image = record.get("image")
    _link_ties(measures)
    _link_vibrato(measures)
    _link_slides(measures)
    _link_arcs(measures)
    _merge_slurs(measures)
    _link_bends(measures)
    _settle_meters(measures)
    _notes_from_heads(measures, score_marks(job_dir))
    digits: list[list[np.ndarray]] = []
    rests: list[bool] = []
    for record in library:
        clean = cv2.imread(str(job_dir / record["image"]), cv2.IMREAD_GRAYSCALE)
        spacing = float(record["spacing"])
        top = float(record["staff_top"]) + float(record["lines"][0])
        bars = (int(record["bars"][0]), int(record["bars"][-1]))
        if clean is None:
            digits.append([])
            rests.append(False)
            continue
        digits.append(multirest_digits(clean, top, spacing, bars))
        geo = geometry(record["staff_top"], tuple(record["lines"]), record["layout"], record["kind"], spacing, bars[0], bars[1])
        rests.append(has_multirest((255.0 - clean.astype(np.float32)) / 255.0, geo))
    _expand_multirests(measures, library, digits, rests)
    return _drop_held_tail(measures)


def estimate_tempo(measures: list[ScoreMeasure]) -> int | None:
    """画面里没印速度时，按每格第一次出现在画面上的时间估速度（每分钟四分音符数，取整）。

    横向滚动、整页翻的谱都是一行（一页）一起换上来，换的时刻比播放到那儿早一个大致固定的提前量，
    所以"时间差 ÷ 拍数差"就是每拍的秒数。每格的起始拍按拍号累加（合并休止按展开后的小节数）。
    同一时刻出现的一批只取最前面那格；长得像的反复小节被合成了一格，时间取的是最早那次、出现帧数特别多，不用。
    每个点和后面 20 个点配对（拍数、时间都增加的），取中位数（Theil–Sen），不怕个别翻页晚了、反复跳回去。

    单拍子（3/4、5/4）和复拍子（6/8、9/8、12/8）混着用的曲子，换拍号时有两种走法：八分音符一样长（按四分音符数），
    或者拍子一样长（复拍子的一拍是附点四分音符）。两种都算一遍，取视频时间对得更齐的那种；
    速度按开头那个拍号的拍子折成四分音符（复拍子的附点四分 = 1.5 个四分音符），和 Guitar Pro 一样。
    点太少、或估出来不像速度（30~300 以外）时返回 None。
    """
    seen = sorted(measure.seen for measure in measures if measure.seen and measure.seconds is not None)
    if not seen:
        return None
    limit = 3 * seen[len(seen) // 2]
    fits = [_fit_tempo(measures, limit, lambda meter: Fraction(1))]
    if len({_compound(measure.meter) for measure in measures}) > 1:
        # 拍子一样长：复拍子按附点四分音符算一拍；开头是复拍子时，折回四分音符要乘 1.5。
        scale = Fraction(3, 2) if _compound(measures[0].meter) else Fraction(1)
        pulse = _fit_tempo(measures, limit, lambda meter: (Fraction(2, 3) if _compound(meter) else Fraction(1)) * scale)
        if pulse is not None and fits[0] is not None and pulse[1] < 0.8 * fits[0][1]:
            fits = [pulse]
    if fits[0] is None:
        return None
    tempo = int(round(60.0 / fits[0][0]))
    return tempo if 30 <= tempo <= 300 else None


def _compound(meter: Fraction | None) -> bool:
    """复拍子：6/8、9/8、12/8 这样（按印出来的拍号：八分、十六分音符为一拍单位，拍数是 3 的倍数且不止 3 拍）。"""
    if meter is None:
        return False
    beats, beat_type = _time(meter)
    return beat_type in (8, 16) and beats % 3 == 0 and beats > 3


def _fit_tempo(measures: list[ScoreMeasure], limit: int, weight) -> tuple[float, float] | None:
    """按 weight（每个四分音符折成几拍，随拍号变）累加起始拍，做 Theil–Sen：返回 (每拍秒数, 散布)。
    散布是两两配对得出的每拍秒数的四分位距 ÷ 中位数：越小，说明这样数拍子和视频时间越对得上。"""
    first: dict[float, float] = {}
    position = Fraction(0)
    for measure in measures:
        start = float(position)
        position += measure.meter * 4 * weight(measure.meter) * (1 + measure.rests_after)
        if measure.seconds is None or measure.seen > limit:
            continue
        first[measure.seconds] = min(first.get(measure.seconds, start), start)
    points = sorted(first.items())
    ratios = sorted(
        (later - earlier) / (end - start)
        for index, (earlier, start) in enumerate(points)
        for later, end in points[index + 1 : index + 21]
        if end > start and later > earlier
    )
    if len(points) < 4 or not ratios:
        return None
    middle = len(ratios) // 2
    per_beat = ratios[middle] if len(ratios) % 2 else (ratios[middle - 1] + ratios[middle]) / 2.0
    spread = (ratios[(3 * len(ratios)) // 4] - ratios[len(ratios) // 4]) / per_beat
    return per_beat, spread


def _drop_held_tail(measures: list[ScoreMeasure], least: int = 3) -> list[ScoreMeasure]:
    """视频结尾常把最后一个音延着多放好几个小节（TAB 上一格格都是同一个带括号的音），原谱到它前面就结束了。
    结尾连着 least 个以上这样的小节就都去掉，最后一个真小节的音也不再延出去。一两个延音小节可能是真的长音，留着。"""

    def held(measure: ScoreMeasure, before: ScoreMeasure) -> bool:
        notes = [note for note in measure.notes if not note.grace]
        last = next((note for note in reversed(before.notes) if not note.rest and not note.grace), None)
        return (
            len(notes) == 1
            and not notes[0].rest
            and notes[0].duration == measure.meter
            and bool(notes[0].frets)
            and all(fret.paren for fret in notes[0].frets)
            and last is not None
            and {(f.string, f.fret) for f in notes[0].frets} == {(f.string, f.fret) for f in last.frets}
        )

    end = len(measures)
    while end > 1 and held(measures[end - 1], measures[end - 2]):
        end -= 1
    if len(measures) - end < least:
        return measures
    for note in reversed(measures[end - 1].notes):
        if not note.rest:
            note.tie_to_next = False
            break
    return measures[:end]


def _opening_key(loaded: list[tuple]) -> int | None:
    """第一个有五线谱的小节开头印的调号（升号数为正、降号数为负）。"""
    for record, images, _tab, staff, clean in loaded:
        if staff is None or not images:
            continue
        image = clean if clean is not None else images[0]
        found = key_signature(image, list(staff.lines), staff.spacing, record["bars"][0], record["bars"][-1])
        return found[0] if found is not None else None
    return None


def _upscaled(job_dir: Path) -> bool:
    """这个作业的谱面是不是放大过才识别的（survey 判断的小谱面模式，记在 manifest 里）。"""
    path = job_dir / "manifest.json"
    if not path.is_file():
        return False
    mode = json.loads(path.read_text(encoding="utf-8")).get("sampling", {}).get("mode") or {}
    return float(mode.get("scale", 1.0)) > 1.0


def _read_measure(
    record: dict,
    images: list[np.ndarray],
    tab,
    staff,
    bank: Bank,
    clean: np.ndarray | None = None,
    start: int | None = None,
    thick: bool = False,
    opening: int | None = None,
) -> list[tuple[int, ScoreMeasure]]:
    """一小节几帧各读一遍，返回不同的读法和各自的票数（参考帧的读法排在最前）。

    品位先按位置投票合成一份；节奏每帧一种读法，最后在 _choose 里结合前后小节挑。
    """
    bars = tuple(record["bars"])
    number = record.get("number")
    image = images[0]
    if tab is None:
        return [(1, ScoreMeasure(number, [], Fraction(1), 4, issues=["没有 TAB"]))]
    readings = [read_frets(frame, tab, bank, bars=bars) for frame in images]
    frets = vote_frets(readings, tab.spacing)
    if clean is not None and clean.shape == image.shape:
        readings.append(read_frets(clean, tab, bank, bars=bars))
        frets = fill_frets(frets, readings[-1], tab.spacing)
    # 没过投票的零星读数留着：五线谱上有音、投票结果里这儿没数字时拿来用（播放高亮盖住了大半帧）。
    reserve = [note for reading in readings for note in reading]
    mark_arcs(image, tab, frets, bars=(bars[0], bars[-1]))
    if clean is not None and clean.shape == image.shape:
        # 参考帧被背景盖住时 "sl."、弧线常看不出来：清洗图（逐像素投票合成）上再找一遍，只补不删。
        mark_arcs(clean, tab, frets, bars=(bars[0], bars[-1]))
    groups = onsets(frets, tab.spacing)
    if staff is None:
        # 纯 TAB（Guitar Pro 那种）：节奏记号画在 TAB 下面。
        results = [read_tab_rhythm(frame, tab, bars, groups) for frame in images]
        out: list[tuple[int, ScoreMeasure]] = []
        for votes, found in _distinct(results, lambda result: tuple((e.duration, e.rest, bool(e.notes)) for e in result.events)):
            notes = [_score_note(event, event.notes) for event in found.events]
            total = sum((note.duration for note in notes), Fraction(0))
            measure = ScoreMeasure(number, notes, total, tab.strings, issues=list(found.issues))
            measure.signature, measure.new_meter = found.meter, found.signature_seen
            out.append((votes, measure))
        return out
    # 五线谱开头的拍号：两个粗体数字常被当成空心符头（全音符）。读出来了才用，读不出的多半不是拍号。
    # 几帧各读一遍，一半以上的帧读出同一个拍号才用。
    readings = [time_signature(((255 - frame) >= 110).astype(np.uint8), staff, bars, reach=9.0, tall=1.0) for frame in images]
    if all(found is None for found, _span in readings) and _header_end_of(record, image) is not None:
        # 行首小节的拍号在谱号、调号后面（四个升号以后离小节线快 10 个线距）：放远一些再找一次。
        readings = [time_signature(((255 - frame) >= 110).astype(np.uint8), staff, bars, reach=13.0, tall=1.0) for frame in images]
    meters = [found for found, _span in readings]
    votes_for = [found for found in meters if found is not None and found <= Fraction(7, 4)]
    meter = None
    if votes_for and len(votes_for) * 2 >= len(meters):
        # 票数一样多时信参考帧（排在最前）。
        meter = max(votes_for, key=lambda found: (votes_for.count(found), found == meters[0]))
    span = next((found_span for found, found_span in readings if found == meter), None)
    margin = 0.5 * staff.spacing
    if meter is not None and span is not None:
        groups = [group for group in groups if not span[0] - margin <= group.x <= span[1] + margin]
    limit = (0, tab.top - 1.2 * tab.spacing)
    # 小节开头印的调号（行首、行中转调）：记下来，升降号那一段不读音。
    key = key_signature(clean if clean is not None and clean.shape == image.shape else image, list(staff.lines), staff.spacing, bars[0], bars[-1])
    if key is None and clean is not None and opening is not None:
        # 清洗图上没认出调号（多帧合成时升号的竖笔偶尔对不齐）：一半以上的原图认出开头那个调号，就是行首重印的调号。
        # 别的调号不这样补：小节线后面紧挨着的八分音符（竖的符干加符尾）在几帧原图里会像个降号。
        found = [key_signature(frame, list(staff.lines), staff.spacing, bars[0], bars[-1]) for frame in images]
        same = [item for item in found if item is not None and item[0] == opening]
        if len(same) * 2 > len(images):
            key = max(same, key=lambda item: item[1])
    if key is not None:
        start = max(start or 0, key[1])
    # 清洗图也当一帧读：只有一两帧原图时，原图上被背景盖住的音清洗图上常常是好的。
    frames = images + ([clean] if clean is not None and clean.shape == image.shape else [])
    results = [read_staff_rhythm(frame, staff, bars, limit=limit, thick=thick) for frame in frames]
    # 跳音的小点、重音记号按帧投票：背景里的碎块、连线的尖头只在一两帧里像个孤立的小点。重音取多数帧认出的那种。
    for attribute, absent in (("staccato", False), ("accent", None)):
        marked = [[(event.x, getattr(event, attribute)) for event in found if getattr(event, attribute)] for found in results]
        for found in results:
            for event in found:
                near = [next((value for x, value in frame if abs(x - event.x) < 0.5 * staff.spacing), None) for frame in marked]
                seen = [value for value in near if value]
                setattr(event, attribute, Counter(seen).most_common(1)[0][0] if len(seen) * 2 > len(results) else absent)
    letters = [bass_attacks(frame, tab, (bars[0], bars[-1])) for frame in frames]
    waves = [vibratos(frame, tab, (bars[0], bars[-1])) for frame in frames]
    holds = [fermatas(frame, staff, (bars[0], bars[-1])) for frame in frames]
    ceiling = staff.bottom + 0.6 * staff.spacing
    arrows = _vote_bends([bends(frame, tab, bank, (bars[0], bars[-1]), ceiling) for frame in frames], tab.spacing)
    out = []
    for votes, events in _distinct(results, lambda found: tuple((e.duration, e.rest) for e in found), positions=lambda found: [e.x for e in found]):
        if meter is not None and span is not None:
            events = [event for event in events if not span[0] - margin <= event.x <= span[1] + margin]
        if start is not None:
            # 行首的谱号、调号那一段：升降号常被认成休止符。
            events = [event for event in events if event.x > start]
        for tip, _amount in arrows:
            # 推弦在五线谱上画成推到的音：被推的音右边一个小符头（常带还原号、升号），正对着 TAB 上的箭头。
            # 从被推的音到箭头尖右边一点，这一段认出来的"休止符"是它们。
            bent = max((event.x for event in events if not event.rest and event.x < tip), default=None)
            if bent is not None:
                events = [event for event in events if not (event.rest and bent < event.x <= tip + 0.5 * staff.spacing)]
        notes, issues = _pair(
            events, groups, tab.spacing, staff, lambda x: stem_event(image, staff, bars, x, limit) or whole_event(image, staff, x, limit), reserve
        )
        _slide_owners(image, staff, notes)
        _tie_arcs(frames, staff, notes)
        _attacks(notes, letters, tab.spacing, trusted=letters[-1] if len(frames) > len(images) else None)
        _vibrato_notes(notes, waves, tab.spacing)
        _fermata_notes(notes, holds, staff.spacing, trusted=holds[-1] if len(frames) > len(images) else None)
        total = sum((note.duration for note in notes), Fraction(0))
        measure = ScoreMeasure(number, notes, total, tab.strings, issues=issues)
        measure.signature, measure.new_meter = meter, meter is not None
        measure.fifths = key[0] if key is not None else None
        measure.bends = [(x, amount, tab.spacing) for x, amount in arrows]
        out.append((votes, measure))
    return out


def _attacks(
    notes: list[ScoreNote], letters: list[list[tuple[float, str]]], spacing: float, trusted: list[tuple[float, str]] | None = None
) -> None:
    """拍弦、勾弦：TAB 上方正对着音的 S / P，一半以上的帧都看到才算（背景里的碎块只在一两帧里像个字母）；
    清洗图（逐像素投票合成的）上看到的直接算：原帧里 P 常和叠在下面的 "H"、弧线粘成一块认不出。
    击弦、勾弦弧线上的 "H"、"P" 印在两个音中间，离音 0.5 个线距以上，不算。"""
    if not letters:
        return
    for note in notes:
        if note.rest or not note.frets:
            continue
        x = note.frets[0].x

        def near(frame: list[tuple[float, str]], kind: str) -> bool:
            return any(abs(lx - x) <= 0.35 * spacing and found == kind for lx, found in frame)

        for kind in ("slap", "pop"):
            seen = sum(near(frame, kind) for frame in letters)
            if seen * 2 > len(letters) or (trusted is not None and near(trusted, kind)):
                note.marks.add(kind)
                break


def _vote_bends(readings: list[list[tuple[float, float]]], spacing: float) -> list[tuple[float, float]]:
    """推弦箭头按位置投票：一半以上的帧在 0.6 个线距以内看到才算，幅度取最多的读法。"""
    pooled = sorted((x, amount) for frame in readings for x, amount in frame)
    clusters: list[list[tuple[float, float]]] = []
    for item in pooled:
        if clusters and item[0] - clusters[-1][-1][0] <= 0.6 * spacing:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    out = []
    for cluster in clusters:
        if len(cluster) * 2 > len(readings):
            amounts = [amount for _x, amount in cluster]
            out.append((float(np.median([x for x, _a in cluster])), max(set(amounts), key=amounts.count)))
    return out


def _link_bends(measures: list[ScoreMeasure]) -> None:
    """推弦归到箭头前面最近的那个音；箭头在这一格里所有音之前（弯线从上一小节跨过来）就归上一小节最后一个音。
    带括号的延音只是推弦那个音延下来的，顺着延音线往前找到真正弹的那个音。"""
    from video2score.refxml import bend_amount

    sounding = [(index, note) for index, measure in enumerate(measures) for note in measure.notes if not note.rest and not note.grace and note.frets]
    for index, measure in enumerate(measures):
        for x, amount, spacing in measure.bends:
            before = [position for position, (owner, note) in enumerate(sounding) if owner == index and note.frets[0].x < x - 0.3 * spacing]
            if before:
                position = before[-1]
            else:
                earlier = [position for position, (owner, _note) in enumerate(sounding) if owner < index]
                if not earlier:
                    continue
                position = earlier[-1]
            while position > 0 and sounding[position][1].tie_from_previous:
                position -= 1
            sounding[position][1].marks.add(f"bend-{bend_amount(str(amount))}")
            # 推弦那道弯线从音的右边伸出去，常被当成滑出、跨小节的连奏滑音：推弦的音和延下来的括号音上的这些去掉，
            # 后面那个音的滑音终点也去掉。
            end = position
            while end + 1 < len(sounding) and sounding[end + 1][1].tie_from_previous:
                end += 1
            for _owner, note in sounding[position : end + 1]:
                note.marks -= {"slide-start", "slur-start", "slide-stop", "slur-stop", "doit", "falloff"} if note is not sounding[position][1] else {"slide-start", "slur-start", "doit", "falloff"}
            if end + 1 < len(sounding):
                sounding[end + 1][1].marks -= {"slide-stop", "slur-stop"}


def _vibrato_notes(notes: list[ScoreNote], waves: list[list[tuple[float, float]]], spacing: float) -> None:
    """揉弦：波浪线左端正对着的那个音（左右 0.8 个线距以内），一半以上的帧都看到才算。"""
    if not waves:
        return
    for note in notes:
        if note.rest or not note.frets:
            continue
        x = note.frets[0].x
        seen = sum(any(abs(left - x) <= 0.8 * spacing for left, _right in frame) for frame in waves)
        if seen * 2 > len(waves):
            note.marks.add("vibrato")


def _fermata_notes(notes: list[ScoreNote], holds: list[list[float]], spacing: float, trusted: list[float] | None = None) -> None:
    """延长记号归横坐标最近的那个音（1.5 个线距以内），一半以上的帧或清洗图上看到才算。"""
    candidates = [x for frame in holds for x in frame]
    for x in candidates:
        seen = sum(any(abs(other - x) <= 0.6 * spacing for other in frame) for frame in holds)
        if seen * 2 <= len(holds) and not (trusted and any(abs(other - x) <= 0.6 * spacing for other in trusted)):
            continue
        sounding = [note for note in notes if not note.rest and note.x is not None]
        nearest = min(sounding, key=lambda note: abs(note.x - x), default=None)
        if nearest is not None and abs(nearest.x - x) <= 1.5 * spacing:
            nearest.marks.add("fermata")


def _link_vibrato(measures: list[ScoreMeasure]) -> None:
    """GP 导出揉弦是一段波浪线：揉弦的音写 start，下一个发声的音写 stop（可以跨小节）。"""
    pending: ScoreNote | None = None
    for measure in measures:
        for note in measure.notes:
            if note.grace or note.rest or not note.frets:
                continue
            if pending is not None:
                note.marks.add("vibrato-stop")
                pending = None
            if "vibrato" in note.marks:
                note.marks.discard("vibrato")
                note.marks.add("vibrato-start")
                pending = note


def _content_starts(loaded: list[tuple]) -> list[int | None]:
    """每一格里行首那一段（谱号、调号）结束的列：用最前面几格切出的行首模板去对（和排版时认行首小节一样），对不上是 None。"""
    from video2score.clean import CleanMeasure
    from video2score.header import content_start, extract_header

    def clean_measure(record: dict, image: np.ndarray) -> CleanMeasure:
        return CleanMeasure(
            image=image,
            spacing=float(record["spacing"]),
            staff_top=float(record["staff_top"]),
            lines=tuple(record["lines"]),
            layout=tuple(tuple(item) for item in record["layout"]),
            kind=record["kind"],
            left_bar=int(record["bars"][0]),
            right_bar=int(record["bars"][-1]),
        )

    usable = [(record, clean if clean is not None else images[0]) for record, images, _tab, _staff, clean in loaded if images]
    try:
        header = extract_header([clean_measure(record, image) for record, image in usable[:3]])
    except (IndexError, ValueError):
        header = None
    if header is None:
        return [None] * len(loaded)
    starts: list[int | None] = []
    for record, images, _tab, _staff, clean in loaded:
        image = clean if clean is not None else (images[0] if images else None)
        try:
            starts.append(content_start(clean_measure(record, image), header) if image is not None else None)
        except (IndexError, ValueError):
            starts.append(None)
    return starts


def _cut_merged(loaded: list[tuple]) -> list[tuple]:
    """截图时一格里混进了后一小节（那条小节线在五线谱上被画面里的东西挡住了，只在 TAB 上看得见）：
    这一格比一般的小节宽得多、TAB 中间有一条贯穿所有弦线的竖线、后一格正好是下一个号，就只留竖线左边那段。"""
    widths = [record["bars"][-1] - record["bars"][0] for record, *_rest in loaded]
    if not widths:
        return loaded
    typical = float(np.median(widths))
    out = []
    for index, (record, images, tab, staff, clean) in enumerate(loaded):
        following = loaded[index + 1][0] if index + 1 < len(loaded) else None
        number = record.get("number")
        if (
            tab is not None
            and images
            and widths[index] > 1.6 * typical
            and number is not None
            and following is not None
            and following.get("number") == number + 1
        ):
            inner = _inner_bar(images[0], tab, record["bars"][0], record["bars"][-1])
            if inner is not None:
                record = {**record, "bars": [record["bars"][0], inner]}
                images = [image[:, : inner + 3] for image in images]
                clean = clean[:, : inner + 3] if clean is not None else None
        out.append((record, images, tab, staff, clean))
    return out


def _inner_bar(image: np.ndarray, tab, left: int, right: int) -> int | None:
    """TAB 第一弦到最后一弦之间、九成以上的行都有墨的细竖线（不挨着两边的小节线）。"""
    y0, y1 = int(round(tab.top)), int(round(tab.bottom))
    margin = int(3 * tab.spacing)
    ink = image[y0 : y1 + 1] < 128
    fraction = ink.mean(axis=0)
    columns = [x for x in range(left + margin, right - margin) if fraction[x] >= 0.9]
    if not columns:
        return None
    groups: list[list[int]] = [[columns[0]]]
    for column in columns[1:]:
        if column - groups[-1][-1] <= 1:
            groups[-1].append(column)
        else:
            groups.append([column])
    thin = [group for group in groups if len(group) <= 4]
    return int(round(np.mean(thin[0]))) if thin else None


def _header_end_of(record: dict, image: np.ndarray) -> int | None:
    """行首小节里谱号、调号结束的列（没有谱号是 None）。"""
    from video2score.clean import CleanMeasure
    from video2score.header import _header_end

    try:
        return _header_end(
            CleanMeasure(
                image=image,
                spacing=float(record["spacing"]),
                staff_top=float(record["staff_top"]),
                lines=tuple(record["lines"]),
                layout=tuple(tuple(item) for item in record["layout"]),
                kind=record["kind"],
                left_bar=int(record["bars"][0]),
                right_bar=int(record["bars"][-1]),
            )
        )
    except (IndexError, ValueError):
        return None


def _distinct(results: list, key, positions=None) -> list[tuple[int, object]]:
    """不同的读法和票数，按第一次出现的先后（参考帧最前）。

    给了 positions（每种读法里各个音的横坐标）时，同一种读法的几帧里挑各音位置离中位数最近的那帧当代表：
    时值序列一样、却有一帧把别处的墨认成了音、漏掉了真的那个，不能因为它是参考帧就用它。"""
    keys = [key(result) for result in results]
    out = []
    for index, value in enumerate(keys):
        if value in keys[:index]:
            continue
        members = [other for other, item in enumerate(keys) if item == value]
        chosen = index
        if positions is not None and len(members) > 2:
            xs = np.array([positions(results[member]) for member in members], float)
            median = np.median(xs, axis=0)
            deviation = [float(np.abs(row - median).sum()) for row in xs]
            chosen = members[int(np.argmin(deviation))]
        out.append((len(members), results[chosen]))
    return out


def _choose(candidates: list[list[tuple[int, ScoreMeasure]]]) -> list[ScoreMeasure]:
    """每小节挑一种读法：票数多的优先；票数一样时，挑时值总和和前后小节一样的（整首的拍号很稳定），
    再挑总和是规整拍号的，最后信参考帧。"""

    def clean(measure: ScoreMeasure) -> bool:
        return measure.meter in _METERS

    first = [max(options, key=lambda item: (item[0], clean(item[1])))[1] for options in candidates]
    chosen: list[ScoreMeasure] = []
    # 前面最近印过的拍号：9/8 这种不在"规整拍号"里，但这一段的小节都该是它。
    current: Fraction | None = None
    for index, options in enumerate(candidates):
        context = {
            neighbour.meter
            for neighbour in (first[index - 1] if index else None, first[index + 1] if index + 1 < len(first) else None)
            if neighbour is not None and clean(neighbour)
        }
        signature = options[0][1].signature
        if signature:
            current = signature
        if current is not None:
            context = {current}
        order = {id(measure): position for position, (_votes, measure) in enumerate(options)}
        # 知道前面印过的拍号时，时值总和对得上它的读法优先于票数：背景很花的小节，几帧原图常一起漏掉同一个音，
        # 只有清洗图读全了。票数最多的读法按拍号修得好（多一个附点、少一个休止这种）就还用它：
        # 少数帧碰巧凑满一小节的读法，内容常常更乱。
        top = max(options, key=lambda item: (item[0], -order[id(item[1])]))[1]
        fits_first = current is not None and top.meter != current and not _repairable(top, current)
        best = max(
            options,
            key=lambda item: ((item[1].meter in context) if fits_first else 0, item[0], item[1].meter in context, clean(item[1]), -order[id(item[1])]),
        )
        chosen.append(best[1])
    return chosen


def _repairable(measure: ScoreMeasure, meter: Fraction) -> bool:
    """这种读法超出拍号、按拍号修正以后正好对上（少了的只能补休止，漏掉的音补不回来，不算修得好）。"""
    if measure.meter <= meter:
        return False
    fixed = _fit_meter([replace(note) for note in measure.notes], meter)
    return fixed is not None and sum((note.duration for note in fixed[0] if not note.grace), Fraction(0)) == meter


def _score_note(event: Event, frets: list[FretNote]) -> ScoreNote:
    """节奏事件 → 音符。不是休止、又没有品位的是延音的后半个音，品位等 _link_ties 从前一个音抄过来。"""
    if event.hollow and frets and all(fret.dead for fret in frets):
        # 闷音的 X 形符头不会是空心的（二分）：X 的两条斜臂被当成了空心环。按符杠数重新定时值。
        base = BASE.get(min(event.beams, 3), Fraction(1, 4))
        event = replace(event, hollow=False, duration=base * (Fraction(3, 2) if event.dots == 1 else Fraction(7, 4) if event.dots >= 2 else 1))
    note = ScoreNote(
        event.duration,
        rest=event.rest,
        frets=list(frets),
        dots=event.dots,
        kind=event.kind,
        tuplet=event.tuplet,
        tuplet_mark=event.tuplet_mark,
    )
    if not event.rest:
        note.tie_from_previous = not frets or any(fret.paren for fret in frets)
    if event.staccato:
        note.marks.add("staccato")
    if event.accent:
        note.marks.add(event.accent)
    if event.stem_up is not None:
        note.stem = "up" if event.stem_up else "down"
    note.beam_next = event.beam_next
    return note


def _link_slides(measures: list[ScoreMeasure]) -> None:
    """TAB 上的滑音斜线变成技巧标记。

    - 两个音之间的斜线上还有弧线：连奏滑音（slide 加 slur）；
    - 两个音之间只有斜线：离前一个音近是前一个音滑出（"/" doit、"\\" falloff），离后一个音近是后一个音滑入（"/" scoop、"\\" plop）；
    - 后面没有音：滑出；一组开头的斜线：滑入。
    """
    notes = [note for measure in measures for note in measure.notes if not note.rest and note.frets]
    for index, note in enumerate(notes):
        for fret in note.frets:
            if fret.slide_in:
                note.marks.add("scoop" if fret.slide_in == "/" else "plop")
            if not fret.slide:
                continue
            arc = fret.slur_next or fret.arc == "start"
            legato = arc or fret.legato
            if fret.slide_next or legato:
                # 连奏滑音可以跨小节线（"sl." 印在斜线上方，后一个音在下一小节开头）。
                following = next((later for later in notes[index + 1 : index + 3] if any(item.string == fret.string for item in later.frets)), None)
                if following is not None and legato:
                    # 连奏滑音写两音之间的滑音；数字下面另画了弧线的再写连线。
                    note.marks.add("slide-start")
                    following.marks.add("slide-stop")
                    if arc:
                        note.marks.add("slur-start")
                        following.marks.add("slur-stop")
                    continue
                if following is not None and fret.slide_x is not None:
                    # 只有一道斜线：五线谱上看得出归谁就按五线谱（滑出画在前一个符头那儿，滑入画在后一个符头那儿）；
                    # 看不出时，离前一个音近是它滑出，离后一个音近是后一个音滑入。
                    target = next(item for item in following.frets if item.string == fret.string)
                    # 五线谱上找不到斜笔时看 TAB 上斜线离谁近；滑出比滑入常见得多，离后一个数字近一倍才算滑入。
                    if note.slide_owner == "in" or (note.slide_owner is None and 2 * (target.left - fret.slide_x) < fret.slide_x - fret.right):
                        following.marks.add("scoop" if fret.slide == "/" else "plop")
                        continue
            note.marks.add("doit" if fret.slide == "/" else "falloff")


def _notes_from_heads(measures: list[ScoreMeasure], marks: Marks) -> None:
    """五线谱上有音、TAB 上没读出数字、又接不上前一个音（占位休止）：按符头在五线谱上的高度推出音高，
    放在同一小节里前后最近那个音所在的弦上（品位 0~24 才用）；整小节时值对得上拍号时才补，对不上的多半是多认的符头。
    前一个音是推弦的：这是推上去以后延着的音（GP 在 TAB 上不写数字，五线谱上画推到的音高），接成推弦那个音的延音。"""
    keys = _measure_keys(measures, marks.key or 0)
    previous_sounding: dict[int, ScoreNote | None] = {}
    last: ScoreNote | None = None
    for measure in measures:
        for note in measure.notes:
            previous_sounding[id(note)] = last
            if not note.rest and not note.grace and note.frets:
                last = note
    for measure, fifths in zip(measures, keys):
        total = sum((note.duration for note in measure.notes if not note.grace), Fraction(0))
        if total != measure.meter or not measure.notes:
            continue
        strings = measure.strings
        tuning = marks.tuning if marks.tuning and len(marks.tuning) == strings else TUNINGS.get(strings, TUNINGS[4])
        bass = min(tuning) < 36
        position = measures.index(measure)
        neighbours = [
            fret
            for other in measures[max(0, position - 1) : position + 2]
            for item in other.notes
            if not item.rest
            for fret in item.frets
            if not fret.dead
        ]
        for index, note in enumerate(measure.notes):
            if not (note.placeholder and note.rest and note.staff_y is not None):
                continue
            before = previous_sounding.get(id(note))
            if before is not None and any(mark.startswith("bend-") for mark in before.marks):
                note.rest, note.placeholder = False, False
                note.frets = [replace(fret, paren=True, slide=None, slide_next=False) for fret in before.frets]
                note.tie_from_previous, before.tie_to_next = True, True
                note.kind = note.kind if note.kind else _kind(note.duration)
                continue
            order = sorted(range(len(measure.notes)), key=lambda other: (abs(other - index), other > index))
            near = [measure.notes[other].frets[0].string for other in order if other != index and measure.notes[other].frets and not measure.notes[other].rest]
            candidates = ([near[0]] if near else []) + [string for string in range(strings, 0, -1) if not near or string != near[0]]
            # 前后小节里弹过同一个音高的，照用它的弦和品。
            pitch = None
            for string in range(1, strings + 1):
                fret = _fret_from_head(note.staff_y, string, tuning, fifths, bass)
                if fret is not None:
                    pitch = tuning[string - 1] + fret
                    break
            same = [item for item in neighbours if pitch is not None and tuning[min(item.string, strings) - 1] + item.fret == pitch]
            if same:
                candidates = [same[0].string] + [string for string in candidates if string != same[0].string]
            for string in candidates:
                fret = _fret_from_head(note.staff_y, string, tuning, fifths, bass)
                if fret is not None:
                    note.rest, note.placeholder = False, False
                    note.frets = [FretNote(string, fret, 0.0, 0, 0, 0.5)]
                    note.kind = note.kind if note.kind else _kind(note.duration)
                    measure.issues.append(f"TAB 上没读到的音，按五线谱符头补成 {string} 弦 {fret} 品")
                    break


def _link_arcs(measures: list[ScoreMeasure]) -> None:
    """TAB 数字上搭着的弧线变成击弦、勾弦：从弧线起头的音，沿同一根弦往后走到弧线收尾的音（最多 4 个音）；
    每两个相邻的音，后一个品位高是击弦（hammer-on），低是勾弦（pull-off）。没找到收尾的，只连到下一个音。"""
    notes = [note for measure in measures for note in measure.notes if not note.rest and note.frets]
    for index, note in enumerate(notes):
        start = next((fret for fret in note.frets if fret.arc == "start"), None)
        if start is None:
            # 分开画的弧线：只连到同一根弦上的下一个音。
            linked = next((fret for fret in note.frets if fret.arc_next), None)
            if linked is not None and ({"hammer", "pull", "slide-start"} & note.marks):
                # 已经在一串击勾弦（或滑音）里了：同一道弧线，不另起一段。
                continue
            following = next((later for later in notes[index + 1 : index + 3] if linked and any(item.string == linked.string for item in later.frets)), None)
            if linked is not None and following is not None:
                b = next(item for item in following.frets if item.string == linked.string)
                if b.fret != linked.fret:
                    kind = "hammer" if b.fret > linked.fret else "pull"
                    note.marks.update((kind, "slur-start"))
                    following.marks.update((f"{kind}-stop", "slur-stop"))
            continue
        chain = [note]
        for later in notes[index + 1 : index + 5]:
            fret = next((item for item in later.frets if item.string == start.string), None)
            if fret is None:
                break
            chain.append(later)
            if fret.arc == "stop":
                break
        if not any(item.string == start.string and item.arc == "stop" for item in chain[-1].frets):
            chain = chain[:2]
        if len(chain) < 2:
            continue
        chain[0].marks.add("slur-start")
        chain[-1].marks.add("slur-stop")
        for first, second in zip(chain, chain[1:]):
            a = next(item for item in first.frets if item.string == start.string)
            b = next(item for item in second.frets if item.string == start.string)
            # 弧线下面是滑音（斜线连着）的两个音，交给 _link_slides 写成连奏滑音。
            if a.fret == b.fret or a.slide_next:
                continue
            kind = "hammer" if b.fret > a.fret else "pull"
            first.marks.add(kind)
            second.marks.add(f"{kind}-stop")


def _merge_slurs(measures: list[ScoreMeasure]) -> None:
    """一道弧线跨几个音（装饰音 → 击弦 → 勾弦）：中间的音既收一段又起一段时，合成一段连线。"""
    for measure in measures:
        for note in measure.notes:
            if {"slur-start", "slur-stop"} <= note.marks:
                note.marks -= {"slur-start", "slur-stop"}


def _link_ties(measures: list[ScoreMeasure]) -> None:
    """延音：后半个音没写品位的抄前一个音的；前一个音记上"延到下一个"。可以跨小节。"""
    previous: ScoreNote | None = None
    for measure in measures:
        for note in measure.notes:
            if note.grace:
                # 装饰音后面紧跟带括号的同一个音：括号音是从装饰音延下来的（GP 画成装饰音连到主音的延音线）。
                previous = note
                continue
            if previous is not None and previous.grace and note.tie_from_previous:
                if not note.frets or {(f.string, f.fret) for f in note.frets} != {(f.string, f.fret) for f in previous.frets}:
                    previous = None
            if note.rest:
                previous = None
                continue
            if note.tie_from_previous:
                same_height = (
                    previous is not None
                    and (note.staff_y is None or previous.staff_y is None or abs(note.staff_y - previous.staff_y) < 0.3)
                )
                if previous is None or (not note.frets and not same_height):
                    note.tie_from_previous = False
                    if not note.frets:
                        note.rest = True
                        note.placeholder = True
                        measure.issues.append("有音、TAB 没读到，先用休止占位")
                        # 占位休止多半是多认出的符头（按拍号修正时会去掉）：不打断前后的延音线。
                        continue
                elif not note.frets and note.tie_arc is False:
                    # 五线谱上两个音之间看得到、却没有延音弧线：是同一个音再弹一次，TAB 上的数字没读出来。
                    note.tie_from_previous = False
                    note.frets = [
                        replace(fret, paren=False, slide=None, slide_next=False, slide_x=None, arc=None, arc_next=False, slur_next=False, legato=False)
                        for fret in previous.frets
                    ]
                    measure.issues.append("TAB 没读到、五线谱上没有延音弧线：按前一个音补成重复音")
                else:
                    if not note.frets:
                        note.frets = [replace(fret, paren=True) for fret in previous.frets]
                    previous.tie_to_next = True
            previous = note


def _pair(
    events: list[Event],
    groups: list[Onset],
    spacing: float,
    staff=None,
    find_stem=None,
    reserve: list[FretNote] | None = None,
) -> tuple[list[ScoreNote], list[str]]:
    """节奏事件按横坐标配上 TAB 发声组。

    五线谱上有音、TAB 上没有数字的：GP 常把延音的后半个音在 TAB 上省掉。先当作延音，
    等 _link_ties 看它和前一个音的符头是不是一样高，不一样高才改回休止占位。
    """
    issues: list[str] = []
    # TAB 上小一号的数字是装饰音：不占时值，五线谱上它那个小符头也不算一个节奏事件。
    # 装饰音总是紧挨在后面一个正常大小的音前面（3 个线距以内）；后面没有音的小号数字是被遮了一截的普通数字。
    graces = [
        group
        for group in groups
        if group.notes
        and all(note.small for note in group.notes)
        and any(0 < other.x - group.x <= 3.0 * spacing and not all(note.small for note in other.notes) for other in groups)
    ]
    remaining = [group for group in groups if group not in graces]
    placed: list[tuple[float, ScoreNote]] = []
    for group in graces:
        head = min((event for event in events if not event.rest), key=lambda event: abs(event.x - group.x), default=None)
        if head is not None and abs(head.x - group.x) <= 1.2 * spacing and all(
            abs(other.x - head.x) > abs(group.x - head.x) for other in remaining
        ):
            events = [event for event in events if event is not head]
        placed.append((group.x, ScoreNote(Fraction(0), frets=list(group.notes), kind="eighth", grace=True)))
    for event in events:
        if event.rest:
            # TAB 上同一位置有音（装饰音的小符头、被认成休止的符尾）就不是休止。
            if any(abs(group.x - event.x) <= 0.8 * spacing for group in groups):
                continue
            placed.append((event.x, _score_note(event, [])))
            continue
        match = min(remaining, key=lambda group: abs(group.x - event.x), default=None)
        if match is not None and abs(match.x - event.x) > 1.2 * spacing:
            match = None
        if match is not None:
            remaining.remove(match)
        frets_found = match.notes if match is not None else _from_reserve(reserve or [], event.x, spacing)
        if (
            match is not None
            and frets_found
            and all(fret.dead for fret in frets_found)
            and not event.crossed
            and not event.hollow
            and find_stem is not None
            and event.head_y is not None
        ):
            # TAB 上是闷音 X，五线谱上却认出一个实心符头：多半是符杠末端那一坨被当成了符头（X 符头本身常认不出来）。
            # 按 X 的符干重新量一遍，符头位置差出一个半线距以上就信符干。
            alt = find_stem(match.x)
            if alt is not None and alt.head_y is not None and abs(alt.head_y - event.head_y) > 1.5 * spacing:
                event = alt
        note = _score_note(event, frets_found)
        if staff is not None and event.head_y is not None:
            note.staff_y = (event.head_y - staff.top) / staff.spacing
            if frets_found and all(fret.dead for fret in frets_found) and (event.crossed or event.hollow) and event.stem_up is not None:
                note.staff_y += 0.5 if event.stem_up else -0.5
        placed.append((event.x, note))
    for group in remaining:
        # 五线谱上没认出符头（X 形闷音符头、被背景盖住）：同一位置有符干就按符干补上这个音；
        # 没有符干的，看是不是全音符。
        found = find_stem(group.x) if find_stem is not None else None
        if found is not None and not any(abs(x - found.x) < 0.5 * spacing for x, _note in placed):
            note = _score_note(found, group.notes)
            if staff is not None and found.head_y is not None:
                note.staff_y = (found.head_y - staff.top) / staff.spacing
            placed.append((group.x, note))
            continue
        issues.append(f"x={group.x:.0f} TAB 有音、五线谱没对上：" + "+".join(f"{n.string}弦{n.fret}品" for n in group.notes))
    placed.sort(key=lambda item: item[0])
    for x, note in placed:
        note.x = x
    return [note for _x, note in placed], issues


def _slide_owners(image: np.ndarray, staff, notes: list[ScoreNote]) -> None:
    """TAB 上两个音之间的一道斜线，GP 画滑出（前一个音）和滑入（后一个音）都一样；五线谱上分得开：
    滑出的斜笔从前一个符头的高度出发，滑入的斜笔落到后一个符头的高度（"/" 看下端和上端，"\\" 反过来）。"""
    spacing = staff.spacing
    ink = (image < 128).astype(np.uint8)
    length = max(9, int(round(1.5 * spacing)))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    tall = max(9, int(round(1.2 * spacing)))
    stems = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall)))
    strokes = cv2.subtract(cv2.subtract(ink, lines), stems)
    # 斜笔被谱线切成上下两截：竖着接回去。
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
    sounding = [note for note in notes if not note.rest and not note.grace and note.frets and note.x is not None and note.staff_y is not None]
    for first, second in zip(sounding, sounding[1:]):
        fret = next((item for item in first.frets if item.slide and item.slide_next), None)
        if fret is None:
            continue
        ay = staff.top + first.staff_y * spacing
        by = staff.top + second.staff_y * spacing
        x0, x1 = int(first.x + 0.5 * spacing), int(second.x - 0.3 * spacing)
        y0, y1 = int(min(ay, by) - 1.5 * spacing), int(max(ay, by) + 1.5 * spacing)
        x0, y0 = max(0, x0), max(0, y0)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        window = strokes[y0:y1, x0:x1]
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
        best = None
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            if not (0.3 * spacing <= w <= 2.0 * spacing and 0.3 * spacing <= h <= 2.0 * spacing) or area > 0.5 * w * h:
                continue
            ys, xs = np.nonzero(labels[y : y + h, x : x + w] == index)
            if xs.size < 4 or np.std(xs) == 0:
                continue
            slope = float(np.corrcoef(xs, ys)[0, 1])
            # "/" 往右上（行号变小，相关为负），"\\" 往右下。
            if (fret.slide == "/") != (slope < -0.5) or abs(slope) < 0.5:
                continue
            if best is None or area > best[0]:
                best = (area, y0 + y, y0 + y + h)
        if best is None:
            continue
        _area, top, bottom = best
        if fret.slide == "/":
            from_first, into_second = abs(bottom - ay), abs(top - by)
        else:
            from_first, into_second = abs(top - ay), abs(bottom - by)
        # 两个符头只差一个线距左右时，斜笔两头离两个符头都差不多近：拿不准时按滑出算（9 首答案里滑出 62 个、滑入 15 个），
        # 落到后一个符头上要明显更准（差半个线距以上）才算滑入。
        first.slide_owner = "in" if into_second + 0.6 * spacing < from_first else "out"


def _tie_arcs(images: list[np.ndarray], staff, notes: list[ScoreNote]) -> None:
    """同一小节里两个一样高的相邻音之间，五线谱上有没有延音弧线（记在后一个音的 tie_arc 上）。

    GP 把小节内延音的后半个音在 TAB 上省掉，只在五线谱上画一道弧线；跨小节线的写成括号数字，不用看。
    弧线在符头上方或下方（两边都看），离符头中心 0.3~1.8 个线距：
    抹掉谱线、符干、实心符头后，在那条带里找弯的细墨块（借用 TAB 击勾弦的 _arcs），够宽、在两个音中间。几帧里有一帧看到就算。
    """
    from video2score.tab import _arcs

    spacing = staff.spacing
    sounding = [note for note in notes if not note.rest and not note.grace and note.x is not None and note.staff_y is not None]
    pairs = [(first, second) for first, second in zip(sounding, sounding[1:]) if abs(first.staff_y - second.staff_y) < 0.3]
    if not pairs:
        return
    strokes_list = []
    for image in images:
        # 延音线只有一像素粗，抗锯齿后偏灰：门槛放宽一些。
        ink = (image < 170).astype(np.uint8)
        length = max(9, int(round(1.5 * spacing)))
        # 只抹谱线那几行上的长横线：长延音线的底部也是一段平的长横线，不能一起抹掉。
        # 谱线只有一两像素粗，只抹线上那一行和上下各一行：贴着谱线走的弧线中段比它粗，还能留下一截。
        band = np.zeros_like(ink)
        crossing = np.zeros_like(ink)
        for line in staff.lines:
            row = int(round(line))
            band[max(0, row - 1) : row + 2] = 1
            # 谱线上下两行（谱线那三行以外）左右两列内都有墨：弧线斜着穿过谱线的地方，这几列的谱线留着，弧线才不断。
            if 2 <= row < ink.shape[0] - 2:
                above = cv2.dilate(ink[row - 2 : row - 1], np.ones((1, 5), np.uint8))
                below = cv2.dilate(ink[row + 2 : row + 3], np.ones((1, 5), np.uint8))
                crossing[max(0, row - 1) : row + 2] = above & below
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))) & band & (1 - crossing)
        tall = max(9, int(round(1.2 * spacing)))
        stems = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall)))
        # 实心符头也抹掉（再往外扩一像素）：小的延音弧线紧贴着符头，不抹会连成一块。
        size = max(3, int(round(0.4 * spacing)))
        heads = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        heads = cv2.dilate(heads, np.ones((3, 3), np.uint8))
        strokes = cv2.subtract(cv2.subtract(cv2.subtract(ink, lines), stems), heads)
        strokes_list.append(strokes)
    # 只用第一帧看两个音中间有没有别的竖笔（升降号、别的音的符干）：有的话那儿挤，弧线常画走样（括号里的音、V 形连线），不下结论。
    first_ink = (images[0] < 128).astype(np.uint8)
    tall = max(9, int(round(1.2 * spacing)))
    verticals = cv2.morphologyEx(first_ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall)))
    for first, second in pairs:
        y = staff.top + (first.staff_y + second.staff_y) / 2.0 * spacing
        x0, x1 = int(first.x + 0.2 * spacing), int(second.x - 0.2 * spacing)
        gap = x1 - x0
        between = verticals[
            max(0, int(y - 2.0 * spacing)) : int(y + 2.0 * spacing), max(0, int(first.x + 0.8 * spacing)) : max(0, int(second.x - 0.8 * spacing))
        ]
        if between.size and between.any():
            continue
        found = False
        for strokes in strokes_list:
            x0c, x1c = max(0, x0), min(strokes.shape[1], x1)
            if x1c - x0c < 3:
                continue
            for side in (-1, 1):
                y0 = int(round(y - 1.8 * spacing if side < 0 else y + 0.3 * spacing))
                y1 = int(round(y - 0.3 * spacing if side < 0 else y + 1.8 * spacing))
                y0, y1 = max(0, y0), min(strokes.shape[0], y1)
                if y1 - y0 < 3:
                    continue
                window = strokes[y0:y1, x0c:x1c].astype(bool)
                # 挨得近的两个音之间的弧线只有半个多线距宽：够宽（半个线距、两音间距的三成以上）、居中就算。
                if any(
                    right - left >= max(0.5 * spacing, 0.3 * gap) and 0.2 * gap <= (left + right) / 2.0 <= 0.8 * gap
                    for left, right in _arcs(window, side, spacing, min_width=0.5)
                ):
                    found = True
                    break
            if found:
                break
        second.tie_arc = found


def _from_reserve(reserve: list[FretNote], x: float, spacing: float) -> list[FretNote]:
    """五线谱上这个位置有音、投票结果里没有数字：各帧零星读到的数字里，每根弦取读到最多（按把握加权）、
    至少读到两次的那个品位。"""
    near = [note for note in reserve if abs(note.x - x) <= 0.8 * spacing and not note.small]
    chosen: list[FretNote] = []
    for string in sorted({note.string for note in near}):
        weights: dict[tuple[int, bool], float] = {}
        counts: dict[tuple[int, bool], int] = {}
        for note in near:
            if note.string == string:
                key = (note.fret, note.dead)
                weights[key] = weights.get(key, 0.0) + note.score
                counts[key] = counts.get(key, 0) + 1
        (fret, dead), _weight = max(weights.items(), key=lambda item: item[1])
        # 只在一帧里出现的是背景杂点：至少两次读到同一个品位才用。
        if counts[(fret, dead)] < 2:
            continue
        best = max((note for note in near if note.string == string and note.fret == fret and note.dead == dead), key=lambda note: note.score)
        # 零星读数只取品位：它旁边的斜线、弧线多半也是同一处画面不干净读出来的。
        chosen.append(replace(best, slide=None, slide_next=False, slide_in=None, slide_x=None, arc=None, arc_next=False, slur_next=False, legato=False))
    return chosen


def _settle_meters(measures: list[ScoreMeasure]) -> None:
    """每小节的拍号。不足的补休止。

    谱上印了拍号（GP 每次换拍号都印）就只在印拍号的小节换；印了但没读出来，用这一小节的时值总和。
    第一个读出来的拍号之前（谱头的拍号常挤在谱号、调号后面没读到），以及一个都没读到时：
    时值总和是规整拍号就用它，否则沿用最常见的拍号。
    """
    first = next((index for index, measure in enumerate(measures) if measure.signature), len(measures))
    sums = [measure.meter for measure in measures[:first] if measure.meter in _METERS and measure.notes]
    if not sums:
        sums = [measure.meter for measure in measures if measure.meter in _METERS and measure.notes]
    common = Counter(sums).most_common(1)[0][0] if sums else Fraction(1)
    current = common
    for index, measure in enumerate(measures):
        total = measure.meter
        following = measures[index + 1].meter if index + 1 < len(measures) else None
        if (
            measure.signature
            and measure.signature not in (total, following)
            and (total == current or (following == current and abs(total - measure.signature) > abs(total - current)))
        ):
            # 读到的拍号和这一小节、下一小节的时值都对不上，而时值正好是原来的拍号：多半是背景里的粗笔画，不换拍。
            measure.issues.append(f"读到拍号 {measure.signature}，和时值 {total} 对不上，没采用")
        elif measure.signature:
            current = measure.signature
        elif measure.new_meter and total in _METERS:
            current = total
        elif index < first and total in _METERS and (total == current or total == following or total == common):
            current = total
        measure.meter = current
        if measure.notes and total > current:
            fixed = _fit_meter(measure.notes, current)
            if fixed is not None:
                measure.notes, reason = fixed
                measure.issues.append(f"时值 {total} 超出拍号 {current}，按拍号修正：{reason}")
                total = current
        if measure.notes and total < current:
            measure.notes.extend(_rest_notes(current - total))
            measure.issues.append(f"时值不足 {total}，补了休止")
        elif total > current:
            measure.issues.append(f"时值超出：{total} > {current}")
        if not measure.notes:
            measure.notes.append(ScoreNote(current, rest=True, kind="whole"))


def _fit_meter(notes: list[ScoreNote], meter: Fraction) -> tuple[list[ScoreNote], str] | None:
    """一小节的时值超出拍号：依次试几种改动，正好填满就用。返回改好的音符和改了什么；都不行返回 None。

    1. 去掉占位休止（五线谱上多认出的符头）；
    2. 连着符杠、却读成四分或更长的音改成八分（符杠连着的音不会比八分长）；
    以下几种只在恰好一处能填满时才改：
    3. 去掉一个附点；4. 去掉一个休止；5. 连着三个一样长的音改成三连音；6. 把一个音减半（少数了一道符杠）。
    """

    def total(items: list[ScoreNote]) -> Fraction:
        return sum((note.duration for note in items if not note.grace), Fraction(0))

    main = [index for index, note in enumerate(notes) if not note.grace]
    # 1. 占位休止一起去掉。
    trial = [note for note in notes if not note.placeholder]
    if len(trial) < len(notes) and total(trial) == meter:
        return trial, "去掉五线谱上多认出的音"
    # 2. 符杠连着的长音。
    linked = set()
    for a, b in zip(main, main[1:]):
        if notes[a].beam_next and not notes[a].rest and not notes[b].rest:
            linked.update((a, b))
    long_linked = [index for index in linked if _base(notes[index]) >= Fraction(1, 4)]
    if long_linked:
        trial = [_with_base(note, Fraction(1, 8)) if index in long_linked else note for index, note in enumerate(notes)]
        if total(trial) == meter:
            return trial, "符杠连着的音改成八分"
    # 2b. 看不到符杠（符头、符干被挡住，按 TAB 补出来的）、读成四分以上、又紧挨着八分或十六分音符的音：多半也在那组符杠下。
    short = {"eighth", "16th", "32nd"}
    hidden = [
        index
        for position, index in enumerate(main)
        if notes[index].beam_next is None
        and not notes[index].rest
        and _base(notes[index]) >= Fraction(1, 4)
        and any(
            0 <= other < len(main) and notes[main[other]].kind in short and not notes[main[other]].rest
            for other in (position - 1, position + 1)
        )
    ]
    if hidden:
        trial = [_with_base(note, Fraction(1, 8)) if index in hidden else note for index, note in enumerate(notes)]
        if total(trial) == meter:
            return trial, "被挡住、挨着符杠的音改成八分"
    single: list[tuple[str, list[ScoreNote]]] = []
    for index in main:
        note = notes[index]
        if note.dots:
            single.append(("去掉一个附点", notes[:index] + [replace(note, dots=0, duration=note.duration / (Fraction(3, 2) if note.dots == 1 else Fraction(7, 4)))] + notes[index + 1 :]))
    for index in main:
        if notes[index].rest:
            single.append(("去掉一个休止", notes[:index] + notes[index + 1 :]))
    for first, second, third in zip(main, main[1:], main[2:]):
        group = [notes[first], notes[second], notes[third]]
        if not any(note.rest or note.tuplet for note in group) and len({note.duration for note in group}) == 1 and third == first + 2:
            triplet = [replace(note, duration=note.duration * Fraction(2, 3), tuplet=Fraction(2, 3), tuplet_mark=mark) for note, mark in zip(group, ("start", None, "stop"))]
            single.append(("改成三连音", notes[:first] + triplet + notes[third + 1 :]))
    for index in main:
        if not notes[index].rest:
            single.append(("一个音减半", notes[:index] + [_with_base(notes[index], _base(notes[index]) / 2)] + notes[index + 1 :]))
    for reason in ("去掉一个附点", "去掉一个休止", "改成三连音", "一个音减半"):
        fits = [trial for name, trial in single if name == reason and total(trial) == meter]
        if len(fits) == 1:
            return fits[0], reason
        if fits:
            return None
    return None


def _base(note: ScoreNote) -> Fraction:
    value = note.duration / (note.tuplet or 1)
    return value / (Fraction(3, 2) if note.dots == 1 else Fraction(7, 4) if note.dots >= 2 else 1)


def _with_base(note: ScoreNote, base: Fraction) -> ScoreNote:
    factor = (Fraction(3, 2) if note.dots == 1 else Fraction(7, 4) if note.dots >= 2 else 1) * (note.tuplet or 1)
    return replace(note, duration=base * factor, kind=_kind(base))


def _expand_multirests(
    measures: list[ScoreMeasure],
    library: list[dict],
    digits: list[list[np.ndarray]] | None = None,
    rests: list[bool] | None = None,
) -> None:
    """一格合并休止印的号是 n，下一格是 m，这一格就代表 m-n 个小节的休止。

    开头那格常带谱号、拍号，号读不出来；它若全是休止，就当作从第 1 小节开始。
    最后一格、或者下一格的号没有旁证时，读谱表上方印的粗体小节数（digits：每格的数字字形）。
    前后号推得出小节数的合并休止，它们的字形先学成这首曲子的模板。
    rests 标出图上有合并休止粗横杠的格：开头那格带谱号、拍号时，那些符号可能被读成了音，照样当休止。
    """
    digits = digits or [[] for _ in measures]
    for measure, bar in zip(measures, rests or []):
        if bar:
            measure.notes = [note for note in measure.notes if note.rest] or [ScoreNote(measure.meter, rest=True)]
    unresolved: list[int] = []
    if measures and measures[0].number is None and all(note.rest for note in measures[0].notes):
        measures[0].number = 1
    silent = [all(note.rest for note in measure.notes) for measure in measures]
    # 整小节休止格里长度已经定了的（一小节，或者已经按号展开）：往后推号时才能跨过它。
    settled: set[int] = set()
    for index, measure in enumerate(measures):
        if silent[index]:
            # 全是休止：一格整小节休止。占位的零碎休止（误读的大号数字、拍号）不保留。
            measure.notes = [ScoreNote(measure.meter, rest=True, kind="whole")]
        if index + 1 >= len(measures) or not silent[index]:
            continue
        # 这一格和下一格的号：没读出来的，从前（后）面最近一个有号的格按格数推（中间隔着的都得是普通的一小节，
        # 隔着别的整小节休止就数不清了）。
        number = _number_near(measures, silent, settled, index)
        reference = _next_numbered(measures, silent, index + 1)
        if number is None or reference is None:
            continue
        ref_index, ref_number = reference
        following = ref_number - (ref_index - index - 1)
        if following - number <= 1:
            settled.add(index)
            continue
        # 下一格的号可能读错（见过最后一格读成 511 的）：后面再一个有号的格得比它大（连着几格合并休止时不一定只大 1），
        # 才按它展开；已经是最后一格时，只信小的间隔。
        later = _next_numbered(measures, silent, ref_index + 1)
        if later is not None:
            trusted = later[1] > ref_number
        elif ref_index + 1 < len(measures):
            trusted = False
        else:
            trusted = following - number <= 16
        if trusted and following - number <= 64:
            measure.rests_after = following - number - 1
            settled.add(index)
            if measure.number is None:
                measure.number = number
                measure.issues.append(f"合并休止的号没读出来，按前面的号推成 {number}")
        else:
            unresolved.append(index)
    # 最后一格合并休止没有下一格的号可推。它的号常被结尾的小节线、画面边缘截掉（比如 177 只剩 "77"）：按前面的号推。
    last = len(measures) - 1
    if measures and all(note.rest for note in measures[last].notes) and last < len(digits) and digits[last]:
        number = _number_near(measures, silent, settled, last)
        if number is not None:
            if measures[last].number is None:
                measures[last].number = number
                measures[last].issues.append(f"合并休止的号没读出来，按前面的号推成 {number}")
            unresolved.append(last)
    learned = _learn_rest_digits(measures, digits)
    for index in unresolved:
        count = read_multirest(digits[index], learned) if index < len(digits) else None
        if count is not None:
            measures[index].rests_after = count - 1
        else:
            measures[index].issues.append("合并休止没有展开：下一格的号没有旁证，印的小节数也没读出来")


def _number_near(measures: list[ScoreMeasure], silent: list[bool], settled: set[int], index: int) -> int | None:
    """这一格的号：读出来了就用；没读出来时，往前找最近一个有号的格按格数推。
    中间（含那一格）有长度没定的整小节休止就数不清，返回 None。"""
    if measures[index].number is not None:
        return measures[index].number
    for other in range(index - 1, -1, -1):
        if silent[other] and other not in settled:
            return None
        if measures[other].number is not None:
            return measures[other].number + measures[other].rests_after + (index - other)
    return None


def _next_numbered(measures: list[ScoreMeasure], silent: list[bool], start: int) -> tuple[int, int] | None:
    """从 start 往后第一个有号的格 (下标, 号)；中间隔着整小节休止（长度不知道）就返回 None。"""
    for other in range(start, len(measures)):
        if measures[other].number is not None:
            return other, measures[other].number
        if silent[other]:
            return None
    return None


def _learn_rest_digits(measures: list[ScoreMeasure], digits: list[list[np.ndarray]]) -> dict[str, np.ndarray]:
    """已经按前后号展开的合并休止：小节数已知，每位数字的字形按数字取平均，做成这首曲子的模板。"""
    samples: dict[str, list[np.ndarray]] = {}
    for measure, canvases in zip(measures, digits):
        if measure.rests_after <= 0:
            continue
        text = str(measure.rests_after + 1)
        if len(text) != len(canvases):
            continue
        for char, canvas in zip(text, canvases):
            samples.setdefault(char, []).append(canvas)
    return {char: np.mean(items, axis=0) for char, items in samples.items()}


def _kind(duration: Fraction) -> str:
    return {
        Fraction(1, 1): "whole",
        Fraction(1, 2): "half",
        Fraction(1, 4): "quarter",
        Fraction(1, 8): "eighth",
        Fraction(1, 16): "16th",
        Fraction(1, 32): "32nd",
    }.get(duration, "quarter")


def write_musicxml(
    path: Path,
    measures: list[ScoreMeasure],
    title: str = "",
    tuning: tuple[int, ...] | None = None,
    marks: Marks | None = None,
    artist: str = "",
    rights: str = "",
) -> None:
    """写成 Guitar Pro 8 能导入的 MusicXML：一个声部、两行谱（和 GP 导出、标准答案一样）。

    曲名写 work-title；原唱写 composer（GP 导出也是这样）；出处（视频作者和链接）写 rights，GP 读成版权一栏。

    - 第 1 行五线谱（贝斯低音谱号、吉他高音谱号，都按习惯高八度记谱，`<transpose>` 移回）：符干、符杠、
      闷音的 x 符头、延音、连音、滑音、连线、击勾弦、滑入滑出、跳音、装饰音。
    - 第 2 行 TAB（带调弦）：同样的音再写一遍，`dynamics="0"` 不发声，带弦和品。
    """
    marks = marks or Marks()
    strings = max((measure.strings for measure in measures), default=4)
    tuning = tuning or marks.tuning or TUNINGS.get(strings, TUNINGS[4])
    keys = _measure_keys(measures, marks.key or 0)
    bass = min(tuning) < 36
    name, program = ("Electric Bass", 34) if bass else ("Electric Guitar", 28)
    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="4.0">',
        f"  <work><work-title>{escape(title)}</work-title></work>",
        *(
            ["  <identification>"]
            + ([f'    <creator type="composer">{escape(artist)}</creator>'] if artist else [])
            + ([f"    <rights>{escape(rights)}</rights>"] if rights else [])
            + ["  </identification>"]
            if artist or rights
            else []
        ),
        "  <part-list>",
        f'    <score-part id="P1"><part-name>{name}</part-name>',
        f'      <score-instrument id="P1-I1"><instrument-name>{name}</instrument-name></score-instrument>',
        f'      <midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>{program}</midi-program></midi-instrument>',
        "    </score-part>",
        "  </part-list>",
        '  <part id="P1">',
    ]
    _settle_dead_frets(measures, tuning, keys, bass)
    written: tuple[int, int] | None = None
    written_key: int | None = None
    number = 0
    for measure, fifths in zip(measures, keys):
        time = _time(measure.meter)
        for repeat in range(1 + measure.rests_after):
            # 小节号：识别出的号（比已经写过的大才用），否则接着上一个号。
            if repeat == 0 and measure.number is not None and measure.number > number:
                number = measure.number
            else:
                number += 1
            first = written is None
            out.append(f'    <measure number="{number}">')
            rest_group = measure.rests_after + 1 if repeat == 0 and measure.rests_after else 0
            if time != written or first or rest_group or fifths != written_key:
                changed = time if time != written or first else None
                key = fifths if fifths != written_key or first else None
                out.append(_attributes(changed, tuning, first=first, fifths=key, multirest=rest_group, bass=bass))
                written, written_key = time, fifths
            if first and marks.tempo:
                out.append(
                    '      <direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                    f"<per-minute>{marks.tempo}</per-minute></metronome></direction-type>"
                    f'<staff>1</staff><sound tempo="{marks.tempo}"/></direction>'
                )
            notes = measure.notes if repeat == 0 else []
            if not notes or all(note.rest for note in notes):
                notes = _rest_notes(measure.meter)
            total = sum((note.duration for note in notes if not note.grace), Fraction(0))
            beams = _beams(notes, time)
            stems = _stems(notes, tuning, bass, beams)
            for index, note in enumerate(notes):
                out.extend(_note_xml(note, tuning, fifths=fifths, staff=1, beams=beams[index], stem=stems[index]))
            out.append(f"      <backup><duration>{int(total * 4 * DIVISIONS)}</duration></backup>")
            for note in notes:
                out.extend(_note_xml(note, tuning, fifths=fifths, staff=2))
            out.append("    </measure>")
    out.append("  </part>")
    out.append("</score-partwise>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _attributes(
    time: tuple[int, int] | None,
    tuning: tuple[int, ...],
    first: bool,
    fifths: int | None = 0,
    multirest: int = 0,
    bass: bool = True,
) -> str:
    parts = ["      <attributes>"]
    if first:
        parts.append(f"        <divisions>{DIVISIONS}</divisions>")
    if fifths is not None:
        # 调号只在开头和转调的小节写。
        parts.append(f'        <key number="1"><fifths>{fifths}</fifths></key>')
        parts.append(f'        <key number="2" print-object="no"><fifths>{fifths}</fifths></key>')
    if time is not None:
        # 拍号只在开头和换拍的小节写。
        beats, beat_type = time
        parts.append(f'        <time number="1"><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>')
        parts.append(f'        <time number="2" print-object="no"><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>')
    if first:
        parts.append("        <staves>2</staves>")
        if bass:
            parts.append('        <clef number="1"><sign>F</sign><line>4</line></clef>')
        else:
            parts.append('        <clef number="1"><sign>G</sign><line>2</line></clef>')
        parts.append('        <clef number="2"><sign>TAB</sign><line>5</line></clef>')
        parts.append('        <staff-details number="2">')
        parts.append(f"          <staff-lines>{len(tuning)}</staff-lines>")
        # MusicXML 的 line 1 是最下面那条线（音最低的弦）。
        for line, midi in enumerate(reversed(tuning), start=1):
            # 调弦的黑键写成降号：降半音调弦是 E♭ A♭ D♭ G♭，不写成 D♯ G♯ C♯ F♯（GP 和谱面都这么印）。
            step, alter, octave = _spell(midi, -1)
            alter_xml = f"<tuning-alter>{alter}</tuning-alter>" if alter else ""
            parts.append(
                f'          <staff-tuning line="{line}"><tuning-step>{step}</tuning-step>{alter_xml}'
                f"<tuning-octave>{octave}</tuning-octave></staff-tuning>"
            )
        parts.append("        </staff-details>")
        # 五线谱按习惯高八度记谱，实际音高低八度。
        parts.append(
            '        <transpose number="1"><diatonic>0</diatonic><chromatic>0</chromatic><octave-change>-1</octave-change></transpose>'
        )
    if multirest > 1:
        # GP 里显示成合并休止（每小节照样写休止）。
        parts.append(f'        <measure-style><multiple-rest use-symbols="no">{multirest}</multiple-rest></measure-style>')
    parts.append("      </attributes>")
    return "\n".join(parts)


def _time(meter: Fraction) -> tuple[int, int]:
    """拍号的写法：读到的拍号照印出来的写（6/8 不写成 3/4），推出来的按四分音符写，不整的按八分音符。"""
    if getattr(meter, "beats", None):
        return meter.beats, meter.beat_type
    quarters = meter * 4
    if quarters.denominator == 1:
        return int(quarters), 4
    eighths = meter * 8
    return int(eighths), 8


_REST_PIECES = (
    (Fraction(1), "whole", 0),
    (Fraction(3, 4), "half", 1),
    (Fraction(1, 2), "half", 0),
    (Fraction(3, 8), "quarter", 1),
    (Fraction(1, 4), "quarter", 0),
    (Fraction(3, 16), "eighth", 1),
    (Fraction(1, 8), "eighth", 0),
    (Fraction(1, 16), "16th", 0),
    (Fraction(1, 32), "32nd", 0),
)


def _rest_notes(meter: Fraction) -> list[ScoreNote]:
    """整小节休止拆成带类型的几个休止（5/4 是全休止加四分休止，3/4 是附点二分休止）。"""
    notes: list[ScoreNote] = []
    left = Fraction(meter)
    for value, kind, dots in _REST_PIECES:
        while left >= value:
            notes.append(ScoreNote(value, rest=True, kind=kind, dots=dots))
            left -= value
    return notes or [ScoreNote(Fraction(meter), rest=True, kind="whole")]


_LEVELS = {"eighth": 1, "16th": 2, "32nd": 3}


def _beams(notes: list[ScoreNote], time: tuple[int, int]) -> list[list[tuple[int, str]]]:
    """每个音的符杠（第几道、begin / continue / end / hook）。

    两个相邻的发声的音（装饰音不算，中间可以隔休止）连不连：图上读到了就按图（ScoreNote.beam_next），
    没读到按拍子分组（x/8 拍里三个八分一组，其余一拍一组）。第二、三道杠只连两边都够短的音，单出来的写成小钩。
    """
    main = [index for index, note in enumerate(notes) if not note.grace]
    beats, beat_type = time
    unit = Fraction(3, 8) if beat_type == 8 and beats % 3 == 0 else Fraction(1, 4)
    onsets: dict[int, Fraction] = {}
    position = Fraction(0)
    for index in main:
        onsets[index] = position
        position += notes[index].duration

    def level(index: int) -> int:
        note = notes[index]
        return 0 if note.rest or not note.frets else _LEVELS.get(note.kind, 0)

    links: dict[int, int] = {}
    sounding = [index for index in main if level(index)]
    for a, b in zip(sounding, sounding[1:]):
        if notes[a].beam_next is not None:
            joined = notes[a].beam_next
        else:
            joined = main.index(b) == main.index(a) + 1 and onsets[a] // unit == onsets[b] // unit
        if joined:
            links[a] = b
    back = {b: a for a, b in links.items()}
    out: list[list[tuple[int, str]]] = [[] for _ in notes]
    for index in sounding:
        for number in range(1, level(index) + 1):
            after = index in links and level(links[index]) >= number
            before = index in back and level(back[index]) >= number
            if after and before:
                value = "continue"
            elif after:
                value = "begin"
            elif before:
                value = "end"
            elif number > 1 and (index in links or index in back):
                # GP 导出的小钩一律写 forward hook（组尾的也是），朝哪边由它自己排，照它写。
                # 组尾带延音的音写成 backward hook 时，导入 GP 后延音线不见了。
                value = "forward hook"
            else:
                continue
            out[index].append((number, value))
    return out


def _stems(
    notes: list[ScoreNote],
    tuning: tuple[int, ...],
    bass: bool,
    beams: list[list[tuple[int, str]]],
) -> list[str | None]:
    """五线谱的符干方向：图上读到的照用；没读到的按符头在中线以上朝下、以下朝上；同一组符杠下的音统一朝多数那边。"""
    middle = 50 if bass else 71  # 低音谱号中线 D3、高音谱号中线 B4（记谱音高）
    stems: list[str | None] = []
    for note in notes:
        if note.rest or not note.frets:
            stems.append(None)
        elif note.grace:
            stems.append("up")
        elif note.stem:
            stems.append(note.stem)
        elif note.kind == "whole":
            stems.append(None)
        else:
            highest = max(tuning[min(fret.string, len(tuning)) - 1] + fret.fret + 12 for fret in note.frets)
            stems.append("down" if highest >= middle else "up")
    group: list[int] = []
    for index, items in enumerate(beams):
        value = dict(items).get(1)
        if value in ("begin", "continue", "end"):
            group.append(index)
        if value == "end" and group:
            directions = [stems[i] for i in group if stems[i]]
            if directions:
                common = max(set(directions), key=directions.count)
                for i in group:
                    stems[i] = common
            group = []
    return stems


def _note_xml(
    note: ScoreNote,
    tuning: tuple[int, ...],
    fifths: int = 0,
    staff: int = 2,
    beams: list[tuple[int, str]] | None = None,
    stem: str | None = None,
) -> list[str]:
    """一个音（和弦是几个 <note>）。staff 1 是五线谱（记谱高八度、符干、符杠），staff 2 是 TAB（不发声、弦和品）。"""
    tab = staff == 2
    voice = 2 if tab else 1
    head = '<note dynamics="0">' if tab else "<note>"
    duration = int(note.duration * 4 * DIVISIONS)
    modification = ""
    tuplet = ""
    if note.tuplet:
        modification = (
            f"<time-modification><actual-notes>{note.tuplet.denominator}</actual-notes>"
            f"<normal-notes>{note.tuplet.numerator}</normal-notes></time-modification>"
        )
        if note.tuplet_mark in ("start", "stop"):
            tuplet = f'<tuplet type="{note.tuplet_mark}" number="1"/>'
        elif note.tuplet_mark == "single":
            tuplet = '<tuplet type="start" number="1"/><tuplet type="stop" number="1"/>'
    if note.rest or not note.frets:
        lines = [
            f"      {head}<rest/><duration>{duration}</duration><voice>{voice}</voice>"
            f"<type>{note.kind}</type>" + "<dot/>" * note.dots + modification + f"<staff>{staff}</staff>"
        ]
        if tuplet:
            lines.append(f"        <notations>{tuplet}</notations>")
        lines.append("      </note>")
        return lines
    number = 2 if tab else 1
    ties = ('<tie type="stop"/>' if note.tie_from_previous else "") + ('<tie type="start"/>' if note.tie_to_next else "")
    tied = ('<tied type="stop"/>' if note.tie_from_previous else "") + ('<tied type="start"/>' if note.tie_to_next else "")
    lines: list[str] = []
    for index, fret in enumerate(note.frets):
        midi = tuning[min(fret.string, len(tuning)) - 1] + fret.fret + (0 if tab else 12)
        step, alter, octave = _spell(midi, fifths)
        chord = "<chord/>" if index else ""
        grace = '<grace slash="yes"/>' if note.grace else ""
        alter_xml = f"<alter>{alter}</alter>" if alter else ""
        timing = "" if note.grace else f"<duration>{duration}</duration>"
        stem_xml = "<stem>none</stem>" if tab else (f"<stem>{stem}</stem>" if stem else "")
        notehead = "<notehead>x</notehead>" if fret.dead else ""
        beam_xml = "" if tab or index else "".join(f'<beam number="{level}">{value}</beam>' for level, value in beams or [])
        lines.append(
            f"      {head}{grace}{chord}<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
            f"{timing}{ties}<voice>{voice}</voice><type>{note.kind}</type>"
            + ("" if note.grace else "<dot/>" * note.dots)
            + modification
            + stem_xml
            + notehead
            + f"<staff>{staff}</staff>"
            + beam_xml
        )
        first = index == 0
        inner = tied
        if first:
            inner += tuplet + _mark_xml(note.marks, number)
        technical = _technical_xml(note.marks, number) if first else ""
        if tab:
            technical = f"<string>{fret.string}</string><fret>{fret.fret}</fret>" + technical
        if technical:
            inner += f"<technical>{technical}</technical>"
        if first:
            inner += _ornaments_xml(note.marks, number) + _articulations_xml(note.marks)
            if "fermata" in note.marks:
                inner += '<fermata type="upright"/>'
            # 拍弦、勾弦 GP 写成 <notations> 里的处理指令（MusicXML 没有对应元素），GP 导入时照这个还原。
            for attack in ("slap", "pop"):
                if attack in note.marks:
                    inner += f'<?GP <root><bass-attack type="{attack}"/></root>?>'
        if inner:
            lines.append(f"        <notations>{inner}</notations>")
        if first and "vibrato-start" in note.marks:
            lines.append('        <?GP <root><vibrato type="Slight"/></root>?>')
        if fret.dead:
            lines.append("        <play><mute>straight</mute></play>")
        lines.append("      </note>")
    return lines


def _measure_keys(measures: list[ScoreMeasure], start: int) -> list[int]:
    """每小节用的调号：开头用全曲调号，小节开头印了新调号就从那里换。"""
    keys: list[int] = []
    current = start
    for measure in measures:
        if measure.fifths is not None:
            current = measure.fifths
        keys.append(current)
    return keys


def _settle_dead_frets(measures: list[ScoreMeasure], tuning: tuple[int, ...], keys: list[int] | int, bass: bool) -> None:
    """闷音在 TAB 上只写 X，没有品位：按五线谱上符头的高度推（GP 照这个音高画 x 符头；同一小节前面有一样高的音时
    照抄它的品，临时升降号也就跟着带上了），推不出时用同一小节里同一根弦上离它最近的音。"""
    if isinstance(keys, int):
        keys = [keys] * len(measures)
    for measure, fifths in zip(measures, keys):
        for index, note in enumerate(measure.notes):
            for fret in note.frets:
                if not fret.dead or fret.fret:
                    continue
                guess = None
                if note.staff_y is not None and len(note.frets) == 1:
                    # 同一小节前面同一根弦上、符头一样高的音：临时升降号管到小节末，照抄它的品。
                    same = [
                        item.fret
                        for other in reversed(measure.notes[:index])
                        if other.staff_y is not None and abs(other.staff_y - note.staff_y) < 0.3
                        for item in other.frets
                        if item.string == fret.string and not item.dead
                    ]
                    guess = same[0] if same else _fret_from_head(note.staff_y, fret.string, tuning, fifths, bass)
                if guess is None:
                    order = sorted(range(len(measure.notes)), key=lambda other: (abs(other - index), other > index))
                    nearby = [
                        item.fret
                        for other in order
                        if other != index
                        for item in measure.notes[other].frets
                        if item.string == fret.string and not item.dead
                    ]
                    guess = nearby[0] if nearby else None
                if guess is not None:
                    fret.fret = guess


_LETTERS = "CDEFGAB"
_SHARP_ORDER = "FCGDAEB"


def _fret_from_head(staff_y: float, string: int, tuning: tuple[int, ...], fifths: int, bass: bool) -> int | None:
    """符头在五线谱上的位置（从第一线往下数几个线距）→ 记谱音高 → 实际音高（低八度）→ 这根弦上的品位。"""
    # 第一线：低音谱号 A3，高音谱号 F5（音名序号, 八度）。
    letter_index, octave = (5, 3) if bass else (3, 5)
    diatonic = octave * 7 + letter_index - int(round(staff_y * 2))
    letter, octave = _LETTERS[diatonic % 7], diatonic // 7
    alter = 0
    if fifths > 0 and letter in _SHARP_ORDER[:fifths]:
        alter = 1
    elif fifths < 0 and letter in _SHARP_ORDER[::-1][: -fifths]:
        alter = -1
    written = 12 * (octave + 1) + _STEPS.index(letter) + alter
    fret = written - 12 - tuning[min(string, len(tuning)) - 1]
    return fret if 0 <= fret <= 24 else None


def _mark_xml(marks: set[str], number: int = 1) -> str:
    """两音之间的滑音、击勾弦的连线（MusicXML 的 <slide>、<slur> 成对写起止）。五线谱编号 1，TAB 编号 2。"""
    out = ""
    if "slide-stop" in marks:
        out += f'<slide type="stop" number="{number}" line-type="solid"/>'
    if "slide-start" in marks:
        out += f'<slide type="start" number="{number}" line-type="solid"/>'
    if "slur-stop" in marks:
        out += f'<slur type="stop" number="{number}" placement="above"/>'
    if "slur-start" in marks:
        out += f'<slur type="start" number="{number}" placement="above"/>'
    return out


def _technical_xml(marks: set[str], number: int = 1) -> str:
    """击弦、勾弦：前一个音写 start（带 H / P 字样），后一个音写 stop。推弦写幅度（半音数，GP 谱上标的是全音数）。"""
    out = ""
    for mark in sorted(marks):
        if mark.startswith("bend-"):
            out += f"<bend><bend-alter>{mark[5:]}</bend-alter></bend>"
    for kind, element in (("hammer", "hammer-on"), ("pull", "pull-off")):
        if f"{kind}-stop" in marks:
            out += f'<{element} type="stop" number="{number}"/>'
    for kind, element, text in (("hammer", "hammer-on", "H"), ("pull", "pull-off", "P")):
        if kind in marks:
            out += f'<{element} type="start" number="{number}">{text}</{element}>'
    return out


def _ornaments_xml(marks: set[str], number: int = 1) -> str:
    """揉弦：GP 导出成波浪线，揉弦的音写 start、下一个音写 stop。"""
    items = ""
    if "vibrato-stop" in marks:
        items += f'<wavy-line type="stop" number="{number}"/>'
    if "vibrato-start" in marks:
        items += f'<wavy-line type="start" number="{number}" placement="above"/>'
    return f"<ornaments>{items}</ornaments>" if items else ""


def _articulations_xml(marks: set[str]) -> str:
    """滑入、滑出、跳音写在 <articulations> 里（GP 导入认 scoop / plop / doit / falloff）。"""
    items = [f"<{name}/>" for name in ("scoop", "plop", "doit", "falloff", "staccato", "accent") if name in marks]
    if "strong-accent" in marks:
        items.append('<strong-accent type="up"/>')
    return f"<articulations>{''.join(items)}</articulations>" if items else ""


def _spell(midi: int, fifths: int = 0) -> tuple[str, int, int]:
    """音名：降号调（fifths < 0）里黑键写成降号（B♭ 而不是 A♯），其余写成升号。"""
    if fifths < 0 and _ALTERS[midi % 12]:
        step = _STEPS[(midi + 1) % 12]
        return step, -1, (midi + 1) // 12 - 1
    return _STEPS[midi % 12], _ALTERS[midi % 12], midi // 12 - 1


def summary(measures: list[ScoreMeasure]) -> dict[str, int]:
    notes = sum(len([n for n in m.notes if not n.rest]) for m in measures)
    return {
        "measures": len(measures),
        "notes": notes,
        "measures_with_issues": sum(1 for m in measures if m.issues),
    }
