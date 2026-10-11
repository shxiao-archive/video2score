"""把识别出的 MusicXML（tabscore.write_musicxml 写的、或 GP 导出的那种两行谱）转成 Guitar Pro 文件：

- `.gp`（GP7/8）：zip 包，乐谱在 Content/score.gpif；样式、版面配置用 gp_template 里 GP8 自己存的那几份。
- `.gpx`（GP6）：同样的 score.gpif，装进 GP6 的 BCFS 文件系统，再按 BCFZ 的格式"压缩"（只用不压缩的字面块）。
- `.gp5`（GP5）：用 pyguitarpro 写。

只读 TAB 那一行（弦、品都在上面）：时值、附点、连音、休止、装饰音、延音、闷音、滑音（两音之间、滑入滑出）、
击勾弦、推弦、揉弦、跳音、重音、拍弦勾弦（GP 的处理指令）、延长记号，以及调号、拍号、速度、调弦。
"""

from __future__ import annotations

import html
import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape

from video2score.paths import GP_TEMPLATE
from video2score.refxml import gp_attack, parse_musicxml

TEMPLATE = GP_TEMPLATE
_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_NOTE_VALUES = {"whole": 1, "half": 2, "quarter": 4, "eighth": 8, "16th": 16, "32nd": 32, "64th": 64}
_GPIF_VALUES = {1: "Whole", 2: "Half", 4: "Quarter", 8: "Eighth", 16: "16th", 32: "32nd", 64: "64th"}
GP5_ENCODING = "gbk"
# GPIF 的滑音标记位。
SHIFT_SLIDE, LEGATO_SLIDE, OUT_DOWN, OUT_UP, IN_BELOW, IN_ABOVE = 1, 2, 4, 8, 16, 32


@dataclass
class GpNote:
    string: int  # 1 是最高的那根弦（和 MusicXML、TAB 一样）
    fret: int
    tie_origin: bool = False
    tie_destination: bool = False
    dead: bool = False
    slide: int = 0  # 上面的标记位
    hopo_origin: bool = False
    hopo_destination: bool = False
    vibrato: bool = False
    bend: float = 0.0  # 推到的半音数
    staccato: bool = False
    accent: bool = False  # 重音 ">"
    heavy_accent: bool = False  # 重音 "^"


@dataclass
class GpBeat:
    value: int  # 1 全音符、2 二分、4 四分……
    dots: int = 0
    tuplet: tuple[int, int] | None = None  # (几个音, 占几个音的时值)，三连音是 (3, 2)
    notes: list[GpNote] = field(default_factory=list)
    grace: bool = False
    slap: bool = False
    pop: bool = False
    fermata: bool = False

    @property
    def length(self) -> Fraction:
        """占的时值（全音符为 1）。装饰音不占。"""
        if self.grace:
            return Fraction(0)
        base = Fraction(1, self.value) * (2 - Fraction(1, 2**self.dots))
        if self.tuplet:
            base = base * self.tuplet[1] / self.tuplet[0]
        return base


@dataclass
class GpBar:
    time: tuple[int, int]
    key: int
    beats: list[GpBeat] = field(default_factory=list)


@dataclass
class GpScore:
    title: str
    tuning: list[int]  # 第 1 弦（最高）在前的 MIDI 音高
    tempo: int
    bars: list[GpBar]
    artist: str = ""  # 原唱
    copyright: str = ""  # 出处（视频作者和链接），印在每页页脚

    @property
    def bass(self) -> bool:
        return min(self.tuning) < 36


def read_score(path: Path, title: str | None = None) -> GpScore:
    """读 MusicXML 的 TAB 那一行。没有 TAB 行（只有五线谱）时读不出弦和品，报错。"""
    root = parse_musicxml(path)
    part = root.find("part")
    if part is None:
        raise ValueError("MusicXML 里没有声部")
    name = title if title is not None else (root.findtext("work/work-title") or root.findtext("movement-title") or "")
    artist = next((item.text or "" for item in root.findall("identification/creator") if item.get("type") in ("composer", "artist")), "")
    rights = root.findtext("identification/rights") or ""
    tab_staff: str | None = None
    tuning: list[int] = []
    tempo = 0
    time, key = (4, 4), 0
    divisions = 1
    bars: list[GpBar] = []
    for measure in part.findall("measure"):
        for attributes in measure.findall("attributes"):
            divisions = int(attributes.findtext("divisions") or divisions)
            found = attributes.find("time")
            if found is not None and found.findtext("beats"):
                time = (int(found.findtext("beats")), int(found.findtext("beat-type")))
            fifths = attributes.find("key/fifths")
            if fifths is not None:
                key = int(fifths.text)
            for clef in attributes.findall("clef"):
                if clef.findtext("sign") == "TAB":
                    tab_staff = clef.get("number", "1")
            for details in attributes.findall("staff-details"):
                lines = sorted(details.findall("staff-tuning"), key=lambda item: int(item.get("line", "0")))
                if lines and not tuning:
                    # line 1 是最低的弦：倒过来，第 1 弦（最高）在前。
                    tuning = [_midi(item.findtext("tuning-step"), item.findtext("tuning-alter"), item.findtext("tuning-octave")) for item in reversed(lines)]
        for sound in measure.iter("sound"):
            if not tempo and sound.get("tempo"):
                tempo = int(round(float(sound.get("tempo"))))
        if not tempo:
            minute = measure.findtext(".//metronome/per-minute")
            if minute:
                tempo = int(round(float(minute)))
        bar = GpBar(time, key)
        for note in measure.findall("note"):
            staff = note.findtext("staff") or "1"
            if tab_staff is not None and staff != tab_staff:
                continue
            _add_note(bar, note, divisions)
        bars.append(bar)
    if tab_staff is None or not tuning:
        raise ValueError("MusicXML 里没有 TAB 行（带调弦和弦、品），转不成 Guitar Pro 文件")
    return GpScore(name, tuning, tempo or 120, bars, artist.strip(), rights.strip())


def _midi(step: str | None, alter: str | None, octave: str | None) -> int:
    return 12 * (int(octave or 0) + 1) + _STEPS[step or "C"] + int(float(alter or 0))


def _add_note(bar: GpBar, element: ET.Element, divisions: int) -> None:
    chord = element.find("chord") is not None
    if chord and bar.beats:
        beat = bar.beats[-1]
    else:
        beat = _beat_of(element, divisions)
        bar.beats.append(beat)
    if element.find("rest") is not None:
        return
    technical = element.find("notations/technical")
    if technical is None or technical.findtext("string") is None:
        return
    note = GpNote(int(technical.findtext("string")), int(technical.findtext("fret") or 0))
    note.tie_origin = any(tie.get("type") == "start" for tie in element.findall("tie"))
    note.tie_destination = any(tie.get("type") == "stop" for tie in element.findall("tie"))
    note.dead = element.findtext("notehead") == "x"
    notations = element.find("notations")
    if notations is not None:
        slide_start = any(item.get("type") == "start" for item in notations.findall("slide"))
        slur_start = any(item.get("type") == "start" for item in notations.findall("slur"))
        if slide_start:
            # 两音之间的滑音：同时起了连线（"sl."）的是连奏滑音，否则是普通的换把滑音。
            note.slide |= LEGATO_SLIDE if slur_start else SHIFT_SLIDE
        articulations = notations.find("articulations")
        if articulations is not None:
            for tag, flag in (("scoop", IN_BELOW), ("plop", IN_ABOVE), ("doit", OUT_UP), ("falloff", OUT_DOWN)):
                if articulations.find(tag) is not None:
                    note.slide |= flag
            note.staccato = articulations.find("staccato") is not None
            note.accent = articulations.find("accent") is not None
            note.heavy_accent = articulations.find("strong-accent") is not None
        for tag in ("hammer-on", "pull-off"):
            for item in technical.findall(tag):
                if item.get("type") == "start":
                    note.hopo_origin = True
                elif item.get("type") == "stop":
                    note.hopo_destination = True
        alter = technical.findtext("bend/bend-alter")
        if alter:
            note.bend = float(alter)
        note.vibrato = any(item.get("type") == "start" for item in notations.findall("ornaments/wavy-line"))
        if notations.find("fermata") is not None:
            beat.fermata = True
        attack = gp_attack(notations)
        if attack == "slap":
            beat.slap = True
        elif attack == "pop":
            beat.pop = True
    beat.notes.append(note)


def _beat_of(element: ET.Element, divisions: int) -> GpBeat:
    grace = element.find("grace") is not None
    kind = element.findtext("type")
    dots = len(element.findall("dot"))
    tuplet = None
    modification = element.find("time-modification")
    if modification is not None:
        tuplet = (int(modification.findtext("actual-notes")), int(modification.findtext("normal-notes")))
    if kind in _NOTE_VALUES:
        value = _NOTE_VALUES[kind]
    else:
        # 整小节休止只写了时值：按时值找最接近的音符（附点）。
        length = Fraction(int(element.findtext("duration") or divisions * 4), divisions * 4)
        value, dots = _value_of(length)
    if grace:
        return GpBeat(value if kind in _NOTE_VALUES else 8, 0, None, grace=True)
    return GpBeat(value, dots, tuplet)


def _value_of(length: Fraction) -> tuple[int, int]:
    for value in (1, 2, 4, 8, 16, 32, 64):
        for dots in (0, 1, 2):
            if Fraction(1, value) * (2 - Fraction(1, 2**dots)) == length:
                return value, dots
    return 1, 0


# ---------------------------------------------------------------- GPIF（.gp 和 .gpx 共用）


def _footer_line(text: str) -> str:
    """GP 页脚（Qt 富文本）里的一行小字，靠左。"""
    if not text:
        return ""
    return (
        '<p align="left" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; '
        f"text-indent:0px; font-family:'Times'; font-size:9pt;\">{html.escape(text, quote=False)}</p>"
    )


def gpif(score: GpScore) -> str:
    """GP7/8 的 score.gpif：开头（标题、速度、音轨、调弦）照 GP8 存的模板，后面是小节、拍、音、时值。"""
    head = (TEMPLATE / "score_head.gpif").read_text(encoding="utf-8")
    for key, value in (("{{TITLE}}", score.title), ("{{ARTIST}}", score.artist), ("{{COPYRIGHT}}", score.copyright)):
        head = head.replace(key, value.replace("]]>", "]] >"))
    # 出处印在每页页脚（页码上面一行）。
    head = head.replace("{{SOURCE}}", _footer_line(score.copyright).replace("]]>", "]] >"))
    head = head.replace("%TEMPO%", str(score.tempo))
    # GPIF 的调弦从最低的弦写起。
    head = head.replace("%TUNING%", " ".join(str(pitch) for pitch in reversed(score.tuning)))
    if not score.bass:
        head = (
            head.replace("Electric Bass (finger)", "Clean Guitar")
            .replace("Electric Bass", "Electric Guitar")
            .replace("<ShortName><![CDATA[Bass]]></ShortName>", "<ShortName><![CDATA[E-Gt]]></ShortName>")
            .replace("<Type>electricBass</Type>", "<Type>electricGuitar</Type>")
            .replace("<Program>33</Program>", "<Program>27</Program>")
            .replace("Stringed/Basses/Clean Bass", "Stringed/Electric Guitars/Clean Guitar")
        )
    out = [head, "<MasterBars>"]
    for index, bar in enumerate(score.bars):
        out.append("<MasterBar>")
        sharps = "Sharps" if bar.key >= 0 else "Flats"
        out.append(f"<Key><AccidentalCount>{bar.key}</AccidentalCount><Mode>Major</Mode><TransposeAs>{sharps}</TransposeAs></Key>")
        out.append(f"<Time>{bar.time[0]}/{bar.time[1]}</Time>")
        fermatas = []
        position = Fraction(0)
        for beat in bar.beats:
            if beat.fermata:
                quarters = position * 4
                fermatas.append(f"<Fermata><Type>Medium</Type><Offset>{quarters.numerator}/{quarters.denominator}</Offset><Length>0.5</Length></Fermata>")
            position += beat.length
        if fermatas:
            out.append("<Fermatas>" + "".join(fermatas) + "</Fermatas>")
        out.append(f"<Bars>{index}</Bars>")
        out.append("</MasterBar>")
    out.append("</MasterBars>")
    clef = "F4" if score.bass else "G2"
    out.append("<Bars>")
    out.extend(f'<Bar id="{index}"><Clef>{clef}</Clef><Voices>{index} -1 -1 -1</Voices></Bar>' for index in range(len(score.bars)))
    out.append("</Bars>")
    voices, beats, notes, rhythms = [], [], [], {}
    for index, bar in enumerate(score.bars):
        ids = []
        bar_beats = bar.beats or [GpBeat(*_value_of(Fraction(bar.time[0], bar.time[1])))]
        for beat in bar_beats:
            key = (beat.value, beat.dots, beat.tuplet)
            if key not in rhythms:
                rhythms[key] = len(rhythms)
            note_ids = []
            for note in beat.notes:
                note_ids.append(len(notes))
                notes.append(_note_gpif(len(notes), note, score.tuning, bar.key))
            ids.append(len(beats))
            beats.append(_beat_gpif(len(beats), beat, rhythms[key], note_ids))
        voices.append(f'<Voice id="{index}"><Beats>{" ".join(map(str, ids))}</Beats></Voice>')
    out += ["<Voices>", *voices, "</Voices>", "<Beats>", *beats, "</Beats>", "<Notes>", *notes, "</Notes>", "<Rhythms>"]
    for (value, dots, tuplet), rhythm in rhythms.items():
        extra = f'<AugmentationDot count="{dots}" />' if dots else ""
        if tuplet:
            extra += f'<PrimaryTuplet num="{tuplet[0]}" den="{tuplet[1]}" />'
        out.append(f'<Rhythm id="{rhythm}"><NoteValue>{_GPIF_VALUES[value]}</NoteValue>{extra}</Rhythm>')
    out += ["</Rhythms>", "</GPIF>", ""]
    return "\n".join(out)


def _beat_gpif(index: int, beat: GpBeat, rhythm: int, note_ids: list[int]) -> str:
    parts = [f'<Beat id="{index}">']
    if beat.grace:
        parts.append("<GraceNotes>BeforeBeat</GraceNotes>")
    parts.append(f'<Dynamic>MF</Dynamic><Rhythm ref="{rhythm}" />')
    if note_ids:
        parts.append(f"<Notes>{' '.join(map(str, note_ids))}</Notes>")
    properties = ""
    if beat.slap:
        properties += '<Property name="Slapped"><Enable /></Property>'
    if beat.pop:
        properties += '<Property name="Popped"><Enable /></Property>'
    if properties:
        parts.append(f"<Properties>{properties}</Properties>")
    parts.append("</Beat>")
    return "".join(parts)


def _note_gpif(index: int, note: GpNote, tuning: list[int], key: int) -> str:
    strings = len(tuning)
    midi = tuning[min(note.string, strings) - 1] + note.fret
    parts = [f'<Note id="{index}">']
    if note.tie_origin or note.tie_destination:
        parts.append(f'<Tie origin="{str(note.tie_origin).lower()}" destination="{str(note.tie_destination).lower()}" />')
    if note.vibrato:
        parts.append("<Vibrato>Slight</Vibrato>")
    # GP 的 Accent 按位记：1 跳音，4 重音 "^"，8 重音 ">"。
    accent = (1 if note.staccato else 0) | (4 if note.heavy_accent else 0) | (8 if note.accent else 0)
    if accent:
        parts.append(f"<Accent>{accent}</Accent>")
    parts.append("<InstrumentArticulation>0</InstrumentArticulation><Properties>")
    if note.bend:
        value = note.bend * 50
        parts.append(
            '<Property name="Bended"><Enable /></Property>'
            '<Property name="BendDestinationOffset"><Float>100.000000</Float></Property>'
            f'<Property name="BendDestinationValue"><Float>{value:.6f}</Float></Property>'
            '<Property name="BendMiddleOffset1"><Float>50.000000</Float></Property>'
            '<Property name="BendMiddleOffset2"><Float>50.000000</Float></Property>'
            f'<Property name="BendMiddleValue"><Float>{value / 2:.6f}</Float></Property>'
            '<Property name="BendOriginOffset"><Float>0.000000</Float></Property>'
            '<Property name="BendOriginValue"><Float>0.000000</Float></Property>'
        )
    parts.append(f'<Property name="ConcertPitch">{_pitch(midi, key, 0)}</Property>')
    parts.append(f'<Property name="Fret"><Fret>{note.fret}</Fret></Property>')
    if note.hopo_destination:
        parts.append('<Property name="HopoDestination"><Enable /></Property>')
    if note.hopo_origin:
        parts.append('<Property name="HopoOrigin"><Enable /></Property>')
    parts.append(f'<Property name="Midi"><Number>{midi}</Number></Property>')
    if note.dead:
        parts.append('<Property name="Muted"><Enable /></Property>')
    if note.slide:
        parts.append(f'<Property name="Slide"><Flags>{note.slide}</Flags></Property>')
    # GPIF 的弦从最低的那根数起（0 是最粗的弦）。
    parts.append(f'<Property name="String"><String>{strings - note.string}</String></Property>')
    parts.append(f'<Property name="TransposedPitch">{_pitch(midi, key, 1)}</Property>')
    parts.append("</Properties></Note>")
    return "".join(parts)


def _pitch(midi: int, key: int, octave_shift: int) -> str:
    name = (_FLAT_NAMES if key < 0 else _SHARP_NAMES)[midi % 12]
    octave = midi // 12 + octave_shift
    # GP 的八度号是 MIDI // 12（MIDI 28 的 E 记作 E2）。降号名的 C♭ 不会出现（只用到黑键）。
    return f"<Pitch><Step>{name[0]}</Step><Accidental>{escape(name[1:])}</Accidental><Octave>{octave}</Octave></Pitch>"


# ---------------------------------------------------------------- .gp（GP7/8）


def write_gp(path: Path, score: GpScore) -> None:
    """GP7/8 的 .gp：zip 包。样式、版面配置用模板里 GP8 自己存的。"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("VERSION", (TEMPLATE / "VERSION").read_bytes())
        archive.writestr("meta.json", (TEMPLATE / "meta.json").read_bytes())
        archive.writestr("Content/", b"")
        for name in ("BinaryStylesheet", "LayoutConfiguration", "PartConfiguration", "Preferences.json"):
            archive.writestr(f"Content/{name}", (TEMPLATE / "Content" / name).read_bytes())
        archive.writestr("Content/Stylesheets/", b"")
        for item in sorted((TEMPLATE / "Content" / "Stylesheets").iterdir()):
            archive.writestr(f"Content/Stylesheets/{item.name}", item.read_bytes())
        archive.writestr("Content/score.gpif", gpif(score).encode("utf-8"))


# ---------------------------------------------------------------- .gpx（GP6）

_SECTOR = 0x1000


def write_gpx(path: Path, score: GpScore) -> None:
    """GP6 的 .gpx：几个文件装进 BCFS 文件系统（按 0x1000 字节分扇区），再写成 BCFZ。"""
    files = [
        ("score.gpif", gpif(score).encode("utf-8")),
        ("BinaryStylesheet", (TEMPLATE / "Content" / "BinaryStylesheet").read_bytes()),
        ("PartConfiguration", (TEMPLATE / "Content" / "PartConfiguration").read_bytes()),
        ("LayoutConfiguration", (TEMPLATE / "Content" / "LayoutConfiguration").read_bytes()),
    ]
    path.write_bytes(bcfz(bcfs(files)))


def bcfs(files: list[tuple[str, bytes]]) -> bytes:
    """BCFS：第 0 扇区空着；每个文件一个目录扇区（类型 2、文件名、大小、数据扇区号表，以 0 结尾），后面跟数据扇区。"""
    sectors: list[bytes] = [bytes(_SECTOR)]
    for name, data in files:
        count = max(1, -(-len(data) // _SECTOR))
        entry_index = len(sectors)
        first_data = entry_index + 1
        entry = bytearray(_SECTOR)
        struct.pack_into("<i", entry, 0, 2)
        encoded = name.encode("utf-8")[:127]
        entry[4 : 4 + len(encoded)] = encoded
        struct.pack_into("<i", entry, 0x8C, len(data))
        if 0x94 + 4 * (count + 1) > _SECTOR:
            raise ValueError(f"{name} 太大，一个目录扇区放不下它的扇区表")
        for offset in range(count):
            struct.pack_into("<i", entry, 0x94 + 4 * offset, first_data + offset)
        sectors.append(bytes(entry))
        for offset in range(count):
            chunk = data[offset * _SECTOR : (offset + 1) * _SECTOR]
            sectors.append(chunk + bytes(_SECTOR - len(chunk)))
    return b"BCFS" + b"".join(sectors)


def bcfz(data: bytes) -> bytes:
    """BCFZ：头 + 解压后的长度 + 位流。只用字面块：1 位标志 0、2 位长度（低位在前）、再逐字节 8 位（高位在前）。"""
    bits = _BitWriter()
    for start in range(0, len(data), 3):
        chunk = data[start : start + 3]
        bits.write(0, 1)
        bits.write_reversed(len(chunk), 2)
        for byte in chunk:
            bits.write(byte, 8)
    return b"BCFZ" + struct.pack("<i", len(data)) + bits.getvalue()


class _BitWriter:
    def __init__(self) -> None:
        self._out = bytearray()
        self._byte = 0
        self._count = 0

    def _bit(self, bit: int) -> None:
        self._byte = (self._byte << 1) | (bit & 1)
        self._count += 1
        if self._count == 8:
            self._out.append(self._byte)
            self._byte = self._count = 0

    def write(self, value: int, count: int) -> None:
        for shift in range(count - 1, -1, -1):
            self._bit(value >> shift)

    def write_reversed(self, value: int, count: int) -> None:
        for shift in range(count):
            self._bit(value >> shift)

    def getvalue(self) -> bytes:
        out = bytes(self._out)
        if self._count:
            out += bytes([self._byte << (8 - self._count)])
        return out


# ---------------------------------------------------------------- .gp5（GP5）


def write_gp5(path: Path, score: GpScore) -> None:
    """GP5：用 pyguitarpro 写。GP5 的装饰音是主音上的效果，延长记号没有，略去。"""
    import guitarpro
    from guitarpro import models as gp

    song = gp.Song(title=score.title, artist=score.artist, copyright=score.copyright, tempo=score.tempo)
    # 版权一栏原样印（默认的两行模板会加上 "Copyright ..."、"All Rights Reserved ..."）；第二行空着。
    song.pageSetup.copyright = "%copyright%\n"
    track = song.tracks[0]
    track.name = "Electric Bass" if score.bass else "Electric Guitar"
    track.strings = [gp.GuitarString(number, pitch) for number, pitch in enumerate(score.tuning, start=1)]
    track.channel.instrument = 33 if score.bass else 27
    song.measureHeaders = []
    track.measures = []
    for index, bar in enumerate(score.bars):
        header = gp.MeasureHeader(number=index + 1)
        header.timeSignature.numerator = bar.time[0]
        header.timeSignature.denominator.value = bar.time[1]
        header.keySignature = _gp5_key(gp, bar.key)
        song.measureHeaders.append(header)
        measure = gp.Measure(track, header)
        track.measures.append(measure)
        voice = measure.voices[0]
        pending_grace: GpBeat | None = None
        for beat in bar.beats:
            if beat.grace:
                pending_grace = beat
                continue
            out = gp.Beat(voice)
            out.duration = gp.Duration(value=beat.value, isDotted=beat.dots >= 1)
            if beat.tuplet:
                out.duration.tuplet = gp.Tuplet(enters=beat.tuplet[0], times=beat.tuplet[1])
            if not beat.notes:
                out.status = gp.BeatStatus.rest
            else:
                out.status = gp.BeatStatus.normal
            if beat.slap:
                out.effect.slapEffect = gp.SlapEffect.slapping
            elif beat.pop:
                out.effect.slapEffect = gp.SlapEffect.popping
            for note in beat.notes:
                out.notes.append(_gp5_note(gp, out, note, pending_grace))
            pending_grace = None
            voice.beats.append(out)
        if not voice.beats:
            rest = gp.Beat(voice, status=gp.BeatStatus.rest)
            value, dots = _value_of(Fraction(bar.time[0], bar.time[1]))
            rest.duration = gp.Duration(value=value, isDotted=dots >= 1)
            voice.beats.append(rest)
    # GP5 的文字按系统代码页存（不是 UTF-8）：中文系统上 GP 按 GBK 读。GBK 里没有的字换成问号。
    for name in ("title", "artist", "copyright"):
        setattr(song, name, getattr(song, name).encode(GP5_ENCODING, errors="replace").decode(GP5_ENCODING))
    track.name = track.name.encode(GP5_ENCODING, errors="replace").decode(GP5_ENCODING)
    guitarpro.write(song, str(path), version=(5, 1, 0), encoding=GP5_ENCODING)


def _gp5_note(gp, beat, note: GpNote, grace: GpBeat | None):
    out = gp.Note(beat, value=note.fret, string=note.string, type=gp.NoteType.normal)
    if note.tie_destination:
        out.type = gp.NoteType.tie
    elif note.dead:
        out.type = gp.NoteType.dead
    effect = out.effect
    effect.hammer = note.hopo_origin
    effect.vibrato = note.vibrato
    effect.staccato = note.staccato
    effect.accentuatedNote = note.accent
    effect.heavyAccentuatedNote = note.heavy_accent
    slides = []
    for flag, kind in (
        (SHIFT_SLIDE, gp.SlideType.shiftSlideTo),
        (LEGATO_SLIDE, gp.SlideType.legatoSlideTo),
        (OUT_DOWN, gp.SlideType.outDownwards),
        (OUT_UP, gp.SlideType.outUpwards),
        (IN_BELOW, gp.SlideType.intoFromBelow),
        (IN_ABOVE, gp.SlideType.intoFromAbove),
    ):
        if note.slide & flag:
            slides.append(kind)
    effect.slides = slides
    if note.bend:
        # GP5 的推弦幅度：value 100 是一个全音（每半音 50）；折线上的点以四分之一音为单位（value ÷ 25）。
        value = int(round(note.bend * 50))
        effect.bend = gp.BendEffect(
            type=gp.BendType.bend,
            value=value,
            points=[gp.BendPoint(0, 0), gp.BendPoint(6, value // 25), gp.BendPoint(12, value // 25)],
        )
    if grace is not None:
        grace_note = next((item for item in grace.notes if item.string == note.string), grace.notes[0] if grace.notes else None)
        if grace_note is not None:
            transition = gp.GraceEffectTransition.hammer if grace_note.hopo_origin else (
                gp.GraceEffectTransition.slide if grace_note.slide else gp.GraceEffectTransition.none
            )
            effect.grace = gp.GraceEffect(fret=grace_note.fret, duration=32, isDead=grace_note.dead, transition=transition)
    return out


def _gp5_key(gp, fifths: int):
    for item in gp.KeySignature:
        if item.value == (fifths, 0):
            return item
    return gp.KeySignature.CMajor


# ---------------------------------------------------------------- 入口


WRITERS = {"gp": write_gp, "gpx": write_gpx, "gp5": write_gp5}


def export(musicxml: Path, formats: list[str], out_dir: Path | None = None, stem: str | None = None, title: str | None = None) -> list[Path]:
    """把一份 MusicXML 转成要的几种 Guitar Pro 格式，和它放在一起（或 out_dir 里），返回写出的文件。"""
    wanted = [item for item in formats if item in WRITERS]
    if not wanted:
        return []
    score = read_score(musicxml, title)
    folder = out_dir or musicxml.parent
    name = stem or musicxml.stem
    written = []
    for kind in wanted:
        target = folder / f"{name}.{kind}"
        WRITERS[kind](target, score)
        written.append(target)
    return written

