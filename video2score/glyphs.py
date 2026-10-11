"""冻结的字形模板和字宽：读号、读品位、拍号、连音数字、调弦字母用到的模板事先画好存成 resources/glyphs.npz。

这些模板原本是运行时用本机字体（Times、Cambria、Segoe UI……）现画的，换台电脑（尤其是 mac）字体不全，识别率就会变。
存下来以后每台电脑用的都是同一套，结果一致。`python -m video2score glyphs` 在装齐这些字体的 Windows 上重新生成；
没有这个文件时各处照旧用本机字体现画。
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from video2score.paths import RESOURCES

GLYPHS = RESOURCES / "glyphs.npz"


@lru_cache(maxsize=1)
def _frozen() -> dict[str, np.ndarray] | None:
    if not GLYPHS.is_file():
        return None
    with np.load(GLYPHS, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def frozen_bank(name: str, numeric: bool = False) -> tuple | None:
    """存下来的一套模板：((标签, 20×12 的图), ...)；没有这个文件或这一套时返回 None。"""
    data = _frozen()
    if data is None or f"{name}_labels" not in data:
        return None
    labels = data[f"{name}_labels"]
    glyphs = data[f"{name}_glyphs"]
    return tuple((int(label) if numeric else str(label), glyphs[index].copy()) for index, label in enumerate(labels))


def build() -> dict[str, int]:
    """用本机字体把每套模板画一遍，写进 glyphs.npz；返回每套的个数。"""
    from video2score.marks import _letter_bank_from_fonts
    from video2score.ocr import _generic_bank_from_fonts
    from video2score.rhythm import _tuplet_bank_from_fonts
    from video2score.tab import _tab_bank_from_fonts
    from video2score.tabrhythm import _signature_bank_from_fonts

    banks = {
        "generic": _generic_bank_from_fonts(),
        "tab": _tab_bank_from_fonts(),
        "tuplet": _tuplet_bank_from_fonts(),
        "signature": _signature_bank_from_fonts(),
        "letters": _letter_bank_from_fonts(),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, bank in banks.items():
        arrays[f"{name}_labels"] = np.array([str(label) for label, _glyph in bank])
        arrays[f"{name}_glyphs"] = np.stack([glyph for _label, glyph in bank])
    RESOURCES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GLYPHS, **arrays)
    _frozen.cache_clear()
    # 排版 PDF 用的字宽也存下来（engrave.font_widths 量的本机字体）。
    from video2score.engrave import FONT_WIDTHS, font_widths

    FONT_WIDTHS.write_text(json.dumps(font_widths(), indent=1) + "\n", encoding="utf-8")
    return {name: len(bank) for name, bank in banks.items()}
