"""按小节重新排版：统一线距，首尾相接排成行，行宽不足时略拉伸，再分页写成 PDF。

每行开头放行首（谱号、调号），行首上方印这一行第一小节的号。
"""

from __future__ import annotations

import io
import re
from itertools import groupby
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video2score.clean import CleanMeasure
from video2score.header import Header

# 纵向 A4，按 200 dpi 出图。
DPI = 200
PAGE_W = 1654
PAGE_H = 2339
MARGIN_X = 110
MARGIN_Y = 120
# 排版后的线距（像素）。200 dpi 下约 1.7 mm，接近印刷谱的常用线距。
SPACING = 13.0
# 行宽差多少以内拉伸补齐。再多就保持参差，免得音符被拉变形。
_MAX_STRETCH = 1.18
# 行与行之间的空白（线距倍数）。
_ROW_GAP = 1.2


@dataclass
class _Placed:
    image: np.ndarray
    staff_top: int


def scale_measure(measure: CleanMeasure, start: int | None = None, stretch_left: bool = True) -> _Placed:
    """缩放到统一线距，并把上下没有墨迹的空白裁掉（谱线所在的区域总是保留）。

    start：左边从哪一列开始（小节自带的行首切掉以后）；没给就从左边小节线开始。
    stretch_left 为假时连左边的小节线也不要（紧跟在行首后面的小节）。
    """
    factor = SPACING / measure.spacing
    size = (
        max(1, int(round(measure.image.shape[1] * factor))),
        max(1, int(round(measure.image.shape[0] * factor))),
    )
    image = cv2.resize(measure.image, size, interpolation=cv2.INTER_AREA)
    staff_top = measure.staff_top * factor
    staff_bottom = staff_top + (measure.lines[-1] if measure.lines else 0) * factor
    rows = np.flatnonzero((image < 200).any(axis=1))
    top = int(min(rows[0], staff_top)) if rows.size else int(staff_top)
    bottom = int(max(rows[-1] + 1, staff_bottom + 1)) if rows.size else int(staff_bottom + 1)
    top = max(0, top - 2)
    bottom = min(image.shape[0], bottom + 2)
    # 小节线两侧留的边距排版时要叠掉，这里按左右小节线裁齐。
    left = int(round((start if start is not None else measure.left_bar) * factor))
    if start is None and not stretch_left:
        left += max(2, int(round(0.2 * SPACING)))
    right = int(round(measure.right_bar * factor)) + 1
    return _Placed(image=image[top:bottom, left:right], staff_top=int(round(staff_top - top)))


def scale_header(header: Header) -> _Placed:
    factor = SPACING / header.spacing
    size = (
        max(1, int(round(header.image.shape[1] * factor))),
        max(1, int(round(header.image.shape[0] * factor))),
    )
    image = cv2.resize(header.image, size, interpolation=cv2.INTER_AREA)
    rows = np.flatnonzero((image < 200).any(axis=1))
    top = max(0, int(rows[0]) - 2) if rows.size else 0
    bottom = min(image.shape[0], int(rows[-1]) + 3) if rows.size else image.shape[0]
    return _Placed(image=image[top:bottom], staff_top=int(round(header.staff_top * factor - top)))


def layout_rows(
    measures: list[CleanMeasure],
    width: int = PAGE_W - 2 * MARGIN_X,
    *,
    header: Header | None = None,
    starts: list[int | None] | None = None,
    numbers: list[int | None] | None = None,
    opening: Header | None = None,
) -> list[np.ndarray]:
    """把小节排成一行一行的图。相邻小节共用一条小节线。

    header：每行开头放的谱号、调号；starts：每小节自带行首时内容开始的列（排版时切掉自带的那段）；
    numbers：每小节的号，印在每行开头；opening：只放在第一行最前面的一段（乐器名、调弦字母）。
    """
    starts = starts or [None] * len(measures)
    numbers = numbers or [None] * len(measures)
    head = scale_header(header) if header is not None else None
    head_width = head.image.shape[1] if head is not None else 0
    first = scale_header(opening) if opening is not None else None
    first_width = first.image.shape[1] if first is not None else 0
    rows: list[list[int]] = []
    current: list[int] = []
    used = head_width + first_width
    for index, measure in enumerate(measures):
        item_width = scale_measure(measure, starts[index]).image.shape[1]
        if current and used + item_width - 1 > width:
            rows.append(current)
            current, used = [], head_width
        current.append(index)
        used += item_width - (1 if len(current) > 1 else 0)
    if current:
        rows.append(current)
    images: list[np.ndarray] = []
    for position, row in enumerate(rows):
        placed = [
            scale_measure(measures[index], starts[index], stretch_left=head is None or offset > 0)
            for offset, index in enumerate(row)
        ]
        lead = ([first] if first is not None and position == 0 else []) + ([head] if head is not None else [])
        image = _join_row(lead + placed)
        last = position == len(rows) - 1
        natural = image.shape[1]
        if natural > width:
            image = cv2.resize(image, (width, max(1, int(round(image.shape[0] * width / natural)))), interpolation=cv2.INTER_AREA)
        elif not last and natural * _MAX_STRETCH >= width:
            image = cv2.resize(image, (width, image.shape[0]), interpolation=cv2.INTER_LINEAR)
        offset = head_width + (first_width if position == 0 else 0)
        images.append(_with_number(image, numbers[row[0]], offset))
    return images


def _with_number(row: np.ndarray, number: int | None, x: int) -> np.ndarray:
    """行首上方印小节号（行图上面加一条空白来放）。"""
    band = int(round(1.6 * SPACING))
    canvas = np.full((row.shape[0] + band, row.shape[1]), 255, np.uint8)
    canvas[band:] = row
    if number is not None:
        from PIL import Image, ImageDraw, ImageFont

        picture = Image.fromarray(canvas)
        font = _number_font()
        ImageDraw.Draw(picture).text((max(0, x - 4), 0), str(number), fill=0, font=font)
        canvas = np.array(picture)
    return canvas


def _number_font():
    from PIL import ImageFont

    from video2score.ocr import _FONT_DIRS

    size = int(round(1.4 * SPACING))
    for name in ("times.ttf", "Times New Roman.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"):
        for directory in _FONT_DIRS:
            path = directory / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _join_row(row: list[_Placed]) -> np.ndarray:
    """按谱线对齐拼成一行。取更黑的像素，接缝处的小节线合成一条。"""
    above = max(item.staff_top for item in row)
    below = max(item.image.shape[0] - item.staff_top for item in row)
    total = sum(item.image.shape[1] for item in row) - (len(row) - 1)
    canvas = np.full((above + below, total), 255, np.uint8)
    x = 0
    for item in row:
        y = above - item.staff_top
        height, item_width = item.image.shape
        region = canvas[y : y + height, x : x + item_width]
        np.minimum(region, item.image, out=region)
        x += item_width - 1
    return canvas


def paginate(rows: list[np.ndarray]) -> list[np.ndarray]:
    """把行依次排进 A4 页面。"""
    pages: list[np.ndarray] = []
    page = _blank_page()
    y = MARGIN_Y
    gap = int(round(SPACING * _ROW_GAP))
    for row in rows:
        height, width = row.shape
        if y > MARGIN_Y and y + height > PAGE_H - MARGIN_Y:
            pages.append(page)
            page = _blank_page()
            y = MARGIN_Y
        height = min(height, PAGE_H - MARGIN_Y - y)
        page[y : y + height, MARGIN_X : MARGIN_X + width] = row[:height]
        y += height + gap
    pages.append(page)
    return pages


def _blank_page() -> np.ndarray:
    return np.full((PAGE_H, PAGE_W), 255, np.uint8)


def write_pdf(path: Path, pages: list[np.ndarray], title: str | None = None, artist: str = "", credit: str = "") -> None:
    """每页嵌一张灰度 PNG，铺满 A4。首页左上写曲名、原唱；页脚右边写页码，左边写出处（credit：视频作者和链接，链接可点）。"""
    import pymupdf

    width_pt = PAGE_W * 72.0 / DPI
    height_pt = PAGE_H * 72.0 / DPI
    heading = " — ".join(item for item in (title, artist) if item)
    document = pymupdf.open()
    try:
        for number, page_image in enumerate(pages, start=1):
            page = document.new_page(width=width_pt, height=height_pt)
            ok, encoded = cv2.imencode(".png", page_image)
            if not ok:
                raise RuntimeError("页面图片编码失败。")
            page.insert_image(page.rect, stream=io.BytesIO(encoded.tobytes()).getvalue())
            label = f"{number} / {len(pages)}"
            page.insert_text(
                (width_pt - 36 - pymupdf.get_text_length(label, fontname="helv", fontsize=8), height_pt - 18),
                label,
                fontsize=8,
                fontname="helv",
                color=(0.4, 0.4, 0.4),
            )
            if credit:
                _write_credit(page, credit, (36, height_pt - 18))
            if number == 1 and heading:
                _write_line(page, (36, 28), heading, 9, (0.4, 0.4, 0.4))
        document.save(path, deflate=True, garbage=4)
    finally:
        document.close()


def _runs(text: str) -> list[tuple[str, str]]:
    """按字体分段：西文用 Helvetica，中日文用 PyMuPDF 自带的简体中文字体（它的西文字母是全角宽，不能混用）。"""

    return [(name, "".join(group)) for name, group in groupby(text, key=lambda char: "helv" if ord(char) < 0x100 else "china-s")]


def _text_width(text: str, size: float) -> float:
    import pymupdf

    return sum(pymupdf.get_text_length(run, fontname=name, fontsize=size) for name, run in _runs(text))


def _write_line(page, origin: tuple[float, float], text: str, size: float, color: tuple[float, float, float]) -> None:
    x, y = origin
    for name, run in _runs(text):
        page.insert_text((x, y), run, fontsize=size, fontname=name, color=color)
        x += _text_width(run, size)


def _write_credit(page, text: str, origin: tuple[float, float]) -> None:
    """页脚左边的出处；里面的网址加上链接。"""
    import pymupdf

    size = 7
    _write_line(page, origin, text, size, (0.35, 0.35, 0.35))
    for match in re.finditer(r"https?://\S+", text):
        left = origin[0] + _text_width(text[: match.start()], size)
        rect = pymupdf.Rect(left, origin[1] - size, left + _text_width(match.group(0), size), origin[1] + 2)
        page.insert_link({"kind": pymupdf.LINK_URI, "from": rect, "uri": match.group(0)})
