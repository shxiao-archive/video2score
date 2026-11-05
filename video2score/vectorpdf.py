"""Guitar Pro 导出的矢量 PDF：不渲染成图，直接读文字和线条（reference.py），换成 MusicXML。

符头、休止、拍号、演奏法是 Bravura 字体的文字，品位是普通文字，谱线、小节线、符干、符杠、滑音线、弧线是矢量线条，
读出来就是准的，用不着图片识别里的投票和修正。
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from video2score.reference import RefEvent, RefMeasure, read_reference
from video2score.tab import FretNote
from video2score.tabscore import Marks, ScoreMeasure, ScoreNote, _kind, _link_vibrato, _merge_slurs, write_musicxml


def is_gp_vector(path: Path) -> bool:
    """页面上有 Bravura 字体的文字、线条里找得到谱线：按 GP 导出的矢量谱读。"""
    import pymupdf

    from video2score.reference import _systems

    with pymupdf.open(path) as document:
        if not len(document):
            return False
        page = document[0]
        if not any("Bravura" in font[3] for font in page.get_fonts()):
            return False
        return bool(_systems(page.get_drawings()))


def score_measures(measures: list[RefMeasure]) -> list[ScoreMeasure]:
    """读出的小节 → 写 MusicXML 用的小节。拍号只在印了的小节有，其余沿用前面的。"""
    out: list[ScoreMeasure] = []
    meter = Fraction(1)
    previous: ScoreNote | None = None
    strings = max((measure.strings for measure in measures), default=0) or 4
    for measure in measures:
        if measure.meter is not None:
            meter = measure.meter
        notes: list[ScoreNote] = []
        for event in measure.events:
            note = _note(event)
            if note.tie_from_previous and previous is not None:
                previous.tie_to_next = True
            notes.extend(_graces(event))
            notes.append(note)
            previous = None if note.rest else note
        _mark_tuplets(notes)
        out.append(
            ScoreMeasure(
                number=measure.number,
                notes=notes,
                meter=meter,
                strings=measure.strings or strings,
                rests_after=measure.count - 1,
                signature=measure.meter,
                new_meter=measure.meter is not None,
                fifths=measure.key,
            )
        )
        if measure.count > 1:
            previous = None
    _link_legato(out)
    _merge_slurs(out)
    _link_vibrato(out)
    return out


def write_score(pdf: Path, path: Path, title: str = "", artist: str = "", rights: str = "") -> tuple[list[ScoreMeasure], int | None]:
    """写 MusicXML，返回写进去的小节和速度（没印就是 None）。"""
    measures = read_reference(pdf)
    if not measures:
        raise RuntimeError("PDF 里没有读到小节。")
    tempo = next((measure.tempo for measure in measures if measure.tempo), None)
    marks = Marks(tuning=measures[0].tuning, key=measures[0].key, tempo=round(tempo) if tempo else None)
    scored = score_measures(measures)
    write_musicxml(path, scored, title=title, marks=marks, artist=artist, rights=rights)
    return scored, marks.tempo


def _note(event: RefEvent) -> ScoreNote:
    frets = [_fret(fret, event.x) for fret in event.frets]
    note = ScoreNote(
        event.duration,
        rest=event.rest,
        frets=frets,
        dots=event.dots,
        kind=_kind(_base(event)),
        tuplet=event.tuplet,
        marks={mark for mark in event.marks if not mark.startswith("tie-")},
        x=event.x,
    )
    if not event.rest:
        note.tie_from_previous = any(fret.paren for fret in event.frets) or "tie-stop" in event.marks
    return note


def _graces(event: RefEvent) -> list[ScoreNote]:
    return [ScoreNote(Fraction(0), frets=[_fret(fret, event.x)], kind="eighth", grace=True, marks=set(event.grace_marks)) for fret in event.grace_frets]


def _link_legato(measures: list[ScoreMeasure]) -> None:
    """击弦、勾弦记在前一个音上：下一个发声的音记收尾，两个音之间画连线。"""
    notes = [note for measure in measures for note in measure.notes if not note.rest and note.frets]
    for note, following in zip(notes, notes[1:]):
        for kind in ("hammer", "pull"):
            if kind in note.marks:
                note.marks.add("slur-start")
                following.marks.update((f"{kind}-stop", "slur-stop"))


def _fret(fret, x: float) -> FretNote:
    dead = fret.fret is None
    return FretNote(fret.string, 0 if dead else fret.fret, x, 0, 0, 1.0, paren=fret.paren, dead=dead)


def _base(event: RefEvent | ScoreNote) -> Fraction:
    value = event.duration / (event.tuplet or 1)
    return value / (Fraction(3, 2) if event.dots == 1 else Fraction(7, 4) if event.dots >= 2 else 1)


def _mark_tuplets(notes: list[ScoreNote]) -> None:
    """连音括号的头尾：同一种连音连着的音，凑满一组（时值合计 = 正常音数 × 最短的音）就收尾。"""
    group: list[ScoreNote] = []
    for note in notes:
        if note.grace:
            continue
        if note.tuplet is None or (group and note.tuplet != group[0].tuplet):
            _close(group)
            group = []
        if note.tuplet is None:
            continue
        group.append(note)
        unit = min(_base(item) for item in group)
        if sum((item.duration for item in group), Fraction(0)) == group[0].tuplet.numerator * unit:
            _close(group)
            group = []
    _close(group)


def _close(group: list[ScoreNote]) -> None:
    if len(group) == 1:
        group[0].tuplet_mark = "single"
    elif group:
        group[0].tuplet_mark = "start"
        group[-1].tuplet_mark = "stop"
