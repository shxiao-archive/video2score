"""把识别出的音高写成 MIDI 或带音名的五线谱，也可以把小节图拼成清晰的 PDF。"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image

from video2score.pitches import midi_for_name, pitch_name, position_for_midi


def write_midi(path: Path, pitches: list[int]) -> None:
    """格式 0 的标准 MIDI。每个音是一个四分音符，按给定顺序排列。"""
    if not pitches:
        raise ValueError("没有可写入的音符。")
    track = bytearray()
    for index, pitch in enumerate(pitches):
        if not 0 <= int(pitch) <= 127:
            raise ValueError(f"音高超出 MIDI 范围：{pitch}")
        track += _varlen(0 if index == 0 else 480)
        track += bytes((0x90, int(pitch), 64))
        track += _varlen(480)
        track += bytes((0x80, int(pitch), 0))
    track += _varlen(0)
    track += bytes((0xFF, 0x2F, 0x00))
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
    header += (1).to_bytes(2, "big") + (480).to_bytes(2, "big")
    body = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    path.write_bytes(header + body)


def read_midi_pitches(path: Path) -> list[int]:
    data = Path(path).read_bytes()
    if not data.startswith(b"MThd"):
        raise ValueError("不是标准 MIDI（缺少 MThd）。")
    track_at = data.find(b"MTrk")
    if track_at < 0:
        raise ValueError("MIDI 里没有 MTrk。")
    length = int.from_bytes(data[track_at + 4 : track_at + 8], "big")
    body = data[track_at + 8 : track_at + 8 + length]
    pitches: list[int] = []
    pos = 0
    running = 0

    def read_varlen() -> int:
        nonlocal pos
        value = 0
        while True:
            byte = body[pos]
            pos += 1
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value

    while pos < len(body):
        read_varlen()
        if pos >= len(body):
            break
        status = body[pos]
        if status & 0x80:
            pos += 1
            running = status
        else:
            status = running
        if status == 0xFF:
            if pos >= len(body):
                break
            pos += 1
            meta_len = read_varlen()
            pos += meta_len
            continue
        kind = status & 0xF0
        if kind in (0xC0, 0xD0):
            pos += 1
            continue
        if pos + 1 >= len(body):
            break
        data1 = body[pos]
        data2 = body[pos + 1]
        pos += 2
        if kind == 0x90 and data2 > 0:
            pitches.append(int(data1))
    return pitches


def write_sheet_pdf(path: Path, measures: list[Image.Image]) -> None:
    """把小节按从左到右、从上到下贴到横向 A4。一像素按 180 dpi 显示，只缩小不放大。"""
    import pymupdf

    if not measures:
        raise ValueError("没有可拼的小节。")
    page_w, page_h = 841.89, 595.28
    margin = 28.0
    gap = 4.0
    content_w = page_w - 2 * margin
    content_h = page_h - 2 * margin
    document = pymupdf.open()
    page = None
    page_index = 0
    x = margin
    y = margin
    row_h = 0.0

    def stamp() -> None:
        if page is None:
            return
        page.insert_text(
            (page_w - margin - 20, page_h - 14),
            str(page_index),
            fontsize=9,
            fontname="helv",
            color=(0.35, 0.35, 0.35),
        )

    for measure in measures:
        image = measure.convert("RGB")
        w_pt, h_pt = _sheet_display_size(image.width, image.height, content_w, content_h)
        if page is not None and x > margin and x + w_pt > page_w - margin:
            y += row_h + gap
            x = margin
            row_h = 0.0
        if page is None or y + h_pt > page_h - margin:
            stamp()
            page = document.new_page(width=page_w, height=page_h)
            page_index += 1
            x = margin
            y = margin
            row_h = 0.0
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        page.insert_image(
            pymupdf.Rect(x, y, x + w_pt, y + h_pt),
            stream=buffer.getvalue(),
            alpha=0,
            keep_proportion=True,
        )
        x += w_pt + gap
        row_h = max(row_h, h_pt)
    stamp()
    document.save(path, deflate=True, garbage=4)
    document.close()


def _sheet_display_size(
    width: int,
    height: int,
    content_w: float,
    content_h: float,
) -> tuple[float, float]:
    w_pt = max(1, width) * 72.0 / 180.0
    h_pt = max(1, height) * 72.0 / 180.0
    shrink = 1.0
    if w_pt > content_w:
        shrink = min(shrink, content_w / w_pt)
    if h_pt * shrink > content_h:
        shrink = min(shrink, content_h / h_pt)
    return w_pt * shrink, h_pt * shrink


def write_notation_pdf(path: Path, pitches: list[int]) -> None:
    """用矢量谱线和真正的音名文本画一页，不嵌入视频截图。"""
    import pymupdf

    if not pitches:
        raise ValueError("没有可写入的音符。")
    document = pymupdf.open()
    page = document.new_page(width=640, height=240)
    top = 70
    gap = 12
    for index in range(5):
        y = top + index * gap
        page.draw_line((36, y), (600, y), color=(0, 0, 0), width=1)
    bottom = top + 4 * gap
    for index, pitch in enumerate(pitches):
        position = position_for_midi(int(pitch))
        x = 70 + index * 42
        y = bottom - position * (gap / 2)
        page.draw_circle((x, y), 5, color=(0, 0, 0), fill=(0, 0, 0))
        page.insert_text((x - 10, 190), pitch_name(int(pitch)), fontsize=11, fontname="helv")
    document.save(path)
    document.close()


def read_notation_pdf_pitches(path: Path) -> list[int]:
    import pymupdf

    data = Path(path).read_bytes()
    if not data.startswith(b"%PDF"):
        raise ValueError("不是 PDF。")
    document = pymupdf.open(stream=io.BytesIO(data), filetype="pdf")
    try:
        image_count = sum(len(page.get_images()) for page in document)
        words: list[tuple[float, float, str]] = []
        for page in document:
            for x0, y0, _x1, _y1, token, *_rest in page.get_text("words"):
                if _is_pitch_token(token):
                    words.append((float(y0), float(x0), token))
    finally:
        document.close()
    words.sort()
    names = [token for _y, _x, token in words]
    if not names:
        if image_count:
            raise ValueError("这是嵌入截图的 PDF，不是识别后的乐谱。")
        raise ValueError("PDF 里没有识别出的音名。")
    return [midi_for_name(name) for name in names]


def read_score_pitches(path: Path) -> list[int]:
    """读回 MIDI、带音名的乐谱 PDF，或 Guitar Pro 文件里的音高。"""
    data = Path(path).read_bytes()
    if data.startswith(b"MThd"):
        return read_midi_pitches(path)
    if data.startswith(b"%PDF"):
        return read_notation_pdf_pitches(path)
    if data.startswith(b"PK"):
        return _read_gp_pitches(path)
    raise ValueError(f"无法当作乐谱读取：{path}")


def _read_gp_pitches(path: Path) -> list[int]:
    with zipfile.ZipFile(path) as archive:
        if "Content/score.gpif" not in archive.namelist():
            raise ValueError("Guitar Pro 文件里没有 Content/score.gpif。")
        xml = archive.read("Content/score.gpif").decode("utf-8")
    pitches: list[int] = []
    for token in xml.split("<MidiPitch>")[1:]:
        pitches.append(int(token.split("</MidiPitch>", 1)[0].strip()))
    if not pitches:
        raise ValueError("Guitar Pro 文件里没有音符。")
    return pitches


def _is_pitch_token(token: str) -> bool:
    try:
        midi_for_name(token)
    except ValueError:
        return False
    return True


def _varlen(value: int) -> bytes:
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(chunks))
