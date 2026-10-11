"""低音谱号不适用。这里是高音谱表、下加一线之上的自然音位置。

位置 0 是最下面一条线，音高 E4。往上每走一格（线或间）升一个自然音。
"""

from __future__ import annotations

# 相对 E4 的半音数。只收录五线谱里常见、不靠升降号能写清的音。
_STEPS_FROM_E4 = (0, 1, 3, 5, 7, 8, 10, 12, 13, 15, 17, 19, 20, 22, 24)
_E4 = 64
_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_for_position(position: int) -> int:
    if position < 0 or position >= len(_STEPS_FROM_E4):
        raise ValueError(f"谱位超出范围：{position}")
    return _E4 + _STEPS_FROM_E4[position]


def position_for_midi(midi: int) -> int:
    for index, step in enumerate(_STEPS_FROM_E4):
        if _E4 + step == midi:
            return index
    raise ValueError(f"没有这个自然音：{midi}")


def pitch_name(midi: int) -> str:
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_for_name(name: str) -> int:
    letter = name.strip()
    for midi in range(_E4, _E4 + _STEPS_FROM_E4[-1] + 1):
        if pitch_name(midi) == letter:
            return midi
    raise ValueError(f"无法识别音名：{name}")
