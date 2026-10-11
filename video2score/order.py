"""用小节号整理追踪出来的序列：定号、拆开误并、合并重复、按号排序。

号是额外的依据，不是必需的。读不出号的槽位保持追踪给出的位置，跟着它前面的槽位走。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from video2score.measures import Measure, loosely_same, same_measure
from video2score.ocr import Bank, Reading, adapted_bank, generic_bank, read_measure
from video2score.track import MAX_OBSERVATIONS, Slot

# 挪动位置、合并槽位这类动作，只用这么有把握的号。
CONFIDENT = 0.70
# 同号的两个槽位隔着不超过这么多个，才当作追踪收重了。
_MERGE_REACH = 4
# 号读得这么准时，同号合并不再要求长相一致。
_SURE_MERGE = 0.80
# 挪到号该在的位置时，最多比后面的小节晚出现这么多秒。
_LATE = 10.0
# 比骨架上最大的号大出这么多、后面又没有号可比的，当误读。
_BEYOND = 20


@dataclass
class Summary:
    read: int
    total: int
    moved: int
    merged: int
    split: int
    cleared: int


def arrange(slots: list[Slot]) -> tuple[list[Slot], Summary]:
    if not slots:
        return slots, Summary(0, 0, 0, 0, 0, 0)
    readings = _read_all(slots, generic_bank())
    _vote_all(slots, readings)
    samples = _trusted_samples(slots, readings)
    if samples:
        readings = _read_all(slots, adapted_bank(samples))
        _vote_all(slots, readings)
    slots, split = _split(slots, readings)
    misread_first = _clear_late_first(slots)
    # 先按号排好，追踪收重的两份才会挨在一起，再合并。
    slots, moved, misread = _reorder(slots)
    slots, merged, cleared = _merge(slots)
    slots, absorbed = _absorb(slots, readings)
    slots, dropped = _drop_redundant(slots)
    merged += absorbed + dropped
    _fill_from_neighbours(slots, readings)
    cleared += _clear_outliers(slots)
    read = sum(1 for slot in slots if slot.number is not None)
    return slots, Summary(read, len(slots), moved, merged, split, cleared + misread + misread_first)


# 第 1 小节总在最开头出现；比最早的槽位晚这么多秒还读成 1 的，是截断的号（11、21 只剩 1）。
_FIRST_WITHIN = 3.0


def _clear_late_first(slots: list[Slot]) -> int:
    """读成 1、却是截断的号的清掉：出现得晚；或者和附近一个号以 1 开头、结尾的槽位（17）长得一模一样。"""
    earliest = min(slot.first_seconds for slot in slots)
    cleared = 0
    for index, slot in enumerate(slots):
        if slot.number != 1:
            continue
        nearby = slots[max(0, index - 3) : index] + slots[index + 1 : index + 4]
        twin = any(
            other.number is not None
            and _truncated(1, other.number)
            and same_measure(slot.latest, other.latest)
            for other in nearby
        )
        if twin or slot.first_seconds > earliest + _FIRST_WITHIN:
            slot.number, slot.number_score = None, 0.0
            cleared += 1
    return cleared


def _clear_outliers(slots: list[Slot]) -> int:
    """比前一格大出一截、又接不上后面的号是误读（最后一格的 174 读成 511），清掉，交给前后去补。"""
    cleared = 0
    changed = True
    # 末尾连着几个误读（701、811）时，清掉后面的，前面的才会露出来：反复做到没有可清的。
    while changed:
        changed = False
        for index, slot in enumerate(slots):
            if slot.number is None:
                continue
            before = next((item.number for item in reversed(slots[:index]) if item.number is not None), None)
            after = next((item.number for item in slots[index + 1 :] if item.number is not None), None)
            if before is None or slot.number - before <= 50:
                continue
            if after is None or after < slot.number or after - before > 50:
                slot.number, slot.number_score = None, 0.0
                cleared += 1
                changed = True
    return cleared


def rescue(slots: list[Slot], fragments: list[Slot]) -> tuple[list[Slot], int]:
    """补漏：整理好的序列里号断了档（32 后面直接是 36），到只出现过一两帧的碎片里找。

    高亮在小节间移动、翻页时，同一小节常被拆成几个只有一两帧的碎片，被当成残影丢了。
    号正好落在断档里、出现的时间也在前后两格之间的碎片，同号的合成一个补进去。只增加，不改动已有的。
    """
    if not fragments:
        return slots, 0
    bank = generic_bank()
    found: dict[int, list[Slot]] = {}
    # 结尾接着往下编的号本身就是旁证，门槛低一些（见过读出 0.6997 的）。
    trailing: dict[int, list[Slot]] = {}
    for fragment in fragments:
        number, score = _vote([read_measure(item, bank) for item in fragment.observations])
        if number is not None and score >= CONFIDENT:
            found.setdefault(number, []).append(fragment)
        if number is not None and score >= 0.6:
            trailing.setdefault(number, []).append(fragment)
    result: list[Slot] = []
    added = 0
    for index, slot in enumerate(slots):
        result.append(slot)
        if index + 1 >= len(slots):
            break
        following = slots[index + 1]
        first, last = slot.number, following.number
        if first is None or last is None or last - first <= 1 or last - first > 12 or is_multirest(slot.latest):
            continue
        start = slot.first_seconds - 5.0
        end = max(item.seconds for item in following.observations) + 5.0
        for number in range(first + 1, last):
            candidates = [fragment for fragment in found.get(number, []) if start <= fragment.first_seconds <= end]
            if not candidates:
                continue
            merged = candidates[0]
            for other in candidates[1:]:
                merged.seen += other.seen
                room = MAX_OBSERVATIONS - len(merged.observations)
                merged.observations.extend(other.observations[: max(0, room)])
            merged.number, merged.number_score = number, CONFIDENT
            result.append(merged)
            added += 1
    # 结尾：谱带最后一屏常只露一两帧就淡出（见过最后两小节只完整出现了两帧）。碎片里号正好接着最后一格往下编、
    # 至少看到两帧、出现在最后一格之后不久的，依次补到末尾；号断了就停。
    tail = result[-1] if result else None
    while tail is not None and tail.number is not None and not is_multirest(tail.latest):
        last_seen = max(item.seconds for item in tail.observations)
        candidates = [
            fragment
            for fragment in trailing.get(tail.number + 1, [])
            if fragment.seen >= 2 and tail.first_seconds <= fragment.first_seconds <= last_seen + 5.0
        ]
        if not candidates:
            break
        merged = max(candidates, key=lambda fragment: fragment.seen)
        merged.number, merged.number_score = tail.number + 1, CONFIDENT
        result.append(merged)
        added += 1
        tail = merged
    return result, added


def _read_all(slots: list[Slot], bank: Bank) -> dict[int, list[Reading | None]]:
    return {id(slot): [read_measure(item, bank) for item in slot.observations] for slot in slots}


def _vote_all(slots: list[Slot], readings: dict[int, list[Reading | None]]) -> None:
    for slot in slots:
        slot.number, slot.number_score = _vote(readings[id(slot)])


def _vote(readings: list[Reading | None]) -> tuple[int | None, float]:
    """多数表决。只有一次读数时，它得和所有其他读数都不冲突才算。"""
    found = [item for item in readings if item is not None]
    if not found:
        return None, 0.0
    counts = Counter(item.number for item in found)
    ranked = sorted(
        counts.items(),
        key=lambda pair: (pair[1], sum(item.score for item in found if item.number == pair[0])),
        reverse=True,
    )
    number, votes = ranked[0]
    if votes < 2 and len(counts) > 1:
        return None, 0.0
    if len(ranked) > 1 and ranked[1][1] == votes:
        return None, 0.0
    scores = [item.score for item in found if item.number == number]
    agreement = votes / len(found)
    return number, float(np.mean(scores)) * agreement


def _trusted_samples(
    slots: list[Slot],
    readings: dict[int, list[Reading | None]],
) -> dict[int, list[np.ndarray]]:
    return harvest([slot.number for slot in slots], [readings[id(slot)] for slot in slots])


def harvest(
    numbers: list[int | None],
    readings: list[list[Reading | None]],
) -> dict[int, list[np.ndarray]]:
    """从这段视频里收可信的字形，做专属模板。

    两种来源：
    - 前后号正好连续（差 1）的槽位，它读出的字形；
    - 前后分别是 n-1 和 n+1 的槽位，它只能是 n。读错了也照收，按位数对上。
      通用模板认错的字体（比如 GP 的 8 总被认成 2）就是这样学会的。
    """
    samples: dict[int, list[np.ndarray]] = {}

    def add(number: int, found: list[Reading | None]) -> None:
        text = str(number)
        for item in found:
            if item is None or len(item.canvases) != len(text):
                continue
            for digit, canvas in zip(text, item.canvases):
                samples.setdefault(int(digit), []).append(canvas)

    for index, number in enumerate(numbers):
        before = numbers[index - 1] if index > 0 else None
        after = numbers[index + 1] if index + 1 < len(numbers) else None
        if before is not None and after is not None and after - before == 2:
            add(before + 1, readings[index])
            continue
        if number is None:
            continue
        if before == number - 1 or after == number + 1:
            add(number, [item for item in readings[index] if item is not None and item.number == number])
    return samples


def _split(slots: list[Slot], readings: dict[int, list[Reading | None]]) -> tuple[list[Slot], int]:
    """一个槽位里清楚地读到两个不同的号（各至少两票），是追踪时把几个小节的观测混在了一起。

    按号分组：号已经有别的槽位的，这组观测并给那个槽位；号没有主人的，另立一个槽位。
    """
    owners = {slot.number: slot for slot in slots if slot.number is not None}
    extra: dict[int, list[tuple[object, Reading]]] = {}
    result: list[Slot] = []
    split = 0
    next_id = max(slot.id for slot in slots) + 1
    for slot in slots:
        found = readings[id(slot)]
        counts = Counter(item.number for item in found if item is not None)
        strong = [number for number, votes in counts.most_common() if votes >= 2]
        if len(strong) < 2:
            result.append(slot)
            continue
        split += 1
        keep = slot.number if slot.number in strong else None
        kept_measures = []
        for measure, item in zip(slot.observations, found):
            if item is None or item.number not in strong or item.number == keep:
                kept_measures.append(measure)
                continue
            owner = owners.get(item.number)
            if owner is not None and owner is not slot:
                if len(owner.observations) < MAX_OBSERVATIONS:
                    owner.observations.append(measure)
                continue
            extra.setdefault(item.number, []).append((measure, item))
        slot.observations = kept_measures or slot.observations
        if keep is not None or kept_measures:
            result.append(slot)
        for number in list(extra):
            pairs = extra.pop(number)
            measures = [measure for measure, _item in pairs]
            piece = Slot(
                next_id,
                observations=measures,
                seen=len(measures),
                first_seconds=min(item.seconds for item in measures),
            )
            piece.number = number
            piece.number_score = float(np.mean([item.score for _measure, item in pairs]))
            owners[number] = piece
            next_id += 1
            result.append(piece)
    return result, split


def _drop_redundant(slots: list[Slot]) -> tuple[list[Slot], int]:
    """两个有号槽位 p、q 之间最多只能有 q-p-1 个小节；多出来的无号槽位是收重的，去掉。

    p 若是多小节休止（谱线中间一道粗横杠），它至少占两个号（p、p+1），中间最多只能有 q-p-2 个小节
    （占几个号要等读出它印的小节数才知道："2" 只占两个号，后面那一小节还是空着的）。
    去掉时先去长相和两边对得上的，再去出现帧数少的。
    """
    numbered = [index for index, slot in enumerate(slots) if slot.number is not None]
    drop: set[int] = set()
    if numbered:
        drop.update(_leading_surplus(slots, numbered[0]))
    for left, right in zip(numbered, numbered[1:]):
        run = list(range(left + 1, right))
        if not run:
            continue
        before, after = slots[left], slots[right]
        assert before.number is not None and after.number is not None

        def alike(index: int) -> bool:
            slot = slots[index]
            return loosely_same(before.latest, slot.latest) or loosely_same(after.latest, slot.latest)

        def redundancy(index: int) -> tuple[int, int]:
            return (0 if alike(index) else 1, slots[index].seen)

        if is_multirest(before.latest):
            # 和两边长得一样的是追踪收重的，都去掉；长得不一样的才按 q-p-2 的空位留。
            twins = [index for index in run if alike(index)]
            drop.update(twins)
            run = [index for index in run if index not in twins]
            room = max(0, after.number - before.number - 2)
        else:
            room = after.number - before.number - 1
        surplus = len(run) - room
        if surplus <= 0:
            continue
        drop.update(sorted(run, key=redundancy)[:surplus])
    return [slot for index, slot in enumerate(slots) if index not in drop], len(drop)


def _leading_surplus(slots: list[Slot], first: int) -> list[int]:
    """第一个有号的槽位（号 n）前面最多只能有 n-1 个小节。多出来的无号槽位里，先去和它长得一样的
    （片头就露出来、号被谱号挡住的同一小节：观测并给它），再去出现得晚的。"""
    head = slots[first]
    assert head.number is not None
    run = list(range(first))
    surplus = len(run) - (head.number - 1)
    if surplus <= 0:
        return []

    def redundancy(index: int) -> tuple[int, float]:
        alike = loosely_same(head.latest, slots[index].latest)
        return (0 if alike else 1, -slots[index].first_seconds)

    dropped = sorted(run, key=redundancy)[:surplus]
    for index in dropped:
        slot = slots[index]
        if loosely_same(head.latest, slot.latest):
            head.seen += slot.seen
            head.first_seconds = min(head.first_seconds, slot.first_seconds)
            room = MAX_OBSERVATIONS - len(head.observations)
            head.observations.extend(slot.observations[: max(0, room)])
    return dropped


def is_multirest(measure: Measure) -> bool:
    """多小节休止：第一组谱线中间有一道又粗又长的横杠。"""
    from video2score.measures import PAD_ABOVE

    count = measure.layout[0][0] if measure.layout else 0
    if count != 5 or len(measure.lines) < 5:
        return False
    spacing = measure.spacing
    middle = int(round(PAD_ABOVE * spacing + measure.lines[2]))
    band = measure.darkness[max(0, middle - int(spacing)) : middle + int(spacing) + 1]
    if band.size == 0:
        return False
    ink = band >= 150
    # 粗横杠这几行里，墨迹连续铺开小节宽度的四分之一以上（号数大时小节宽、横杠相对短）。
    thick = 0
    for row in ink:
        runs = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
        longest = int((ends - starts).max()) if starts.size else 0
        if longest >= 0.25 * measure.width:
            thick += 1
    return thick >= 0.4 * spacing


def _fill_from_neighbours(
    slots: list[Slot],
    readings: dict[int, list[Reading | None]],
) -> int:
    """没定出号的槽位，按前后的号补：

    - 读数里恰好有一个落在前后两个号之间，就是它（平票时这样裁决：87 和 822，取 87）；
    - 前后号正好差 2、中间只有它一个，读不出也只能是中间那个号。
    """
    filled = 0
    for index, slot in enumerate(slots):
        if slot.number is not None or index == 0 or index + 1 >= len(slots):
            continue
        before, after = slots[index - 1].number, slots[index + 1].number
        if before is None or after is None:
            continue
        candidates = {
            item.number
            for item in readings.get(id(slot), [])
            if item is not None and item.score >= 0.6 and before < item.number < after
        }
        if len(candidates) == 1:
            slot.number = candidates.pop()
            slot.number_score = 0.6
            filled += 1
        elif not candidates and after - before == 2:
            slot.number = before + 1
            slot.number_score = 0.5
            filled += 1
    return filled


def _merge(slots: list[Slot]) -> tuple[list[Slot], int, int]:
    """号相同、长相也对得上、又挨得近的，是同一个小节被收了两次，合成一个。

    追踪收重的两份总在附近；隔得远的同号多半是误读（48 读成 42），只把把握小的那个清掉号，不合并。
    """
    first: dict[int, tuple[int, Slot]] = {}
    result: list[Slot] = []
    merged = cleared = 0
    for slot in slots:
        if slot.number is None:
            result.append(slot)
            continue
        found = first.get(slot.number)
        if found is None:
            first[slot.number] = (len(result), slot)
            result.append(slot)
            continue
        position, earlier = found
        confident = min(earlier.number_score, slot.number_score) >= CONFIDENT
        near = len(result) - position <= _MERGE_REACH
        # 号是唯一的。挨得近、号又读得很准，长相对不上（音符动画、背景）也是同一个小节。
        sure = min(earlier.number_score, slot.number_score) >= _SURE_MERGE
        if near and ((confident and loosely_same(earlier.latest, slot.latest)) or sure):
            earlier.seen += slot.seen
            room = MAX_OBSERVATIONS - len(earlier.observations)
            earlier.observations.extend(slot.observations[: max(0, room)])
            earlier.number_score = max(earlier.number_score, slot.number_score)
            merged += 1
            continue
        loser = slot if slot.number_score <= earlier.number_score else earlier
        if loser is earlier:
            first[slot.number] = (len(result), slot)
        loser.number = None
        loser.number_score = 0.0
        cleared += 1
        result.append(slot)
    return result, merged, cleared


def _absorb(
    slots: list[Slot],
    readings: dict[int, list[Reading | None]],
) -> tuple[list[Slot], int]:
    """没定出号的槽位，若有一次观测清楚读到了紧挨着的槽位的号，就是那个小节收重的一份，并进去。

    这种槽位多半只看到过号被截掉、被遮住的几帧，表决不出号。
    """
    result: list[Slot] = []
    absorbed = 0
    owners = {slot.number: slot for slot in slots if slot.number is not None}
    for index, slot in enumerate(slots):
        if slot.number is not None:
            result.append(slot)
            continue
        # 挨着就是很强的证据，读数把握低一点、多读出一位（10 读成 101）也认。
        heard = {
            item.number
            for item in readings.get(id(slot), [])
            if item is not None and item.score >= 0.6
        }
        neighbours = [result[-1]] if result else []
        if index + 1 < len(slots):
            neighbours.append(slots[index + 1])
        # 号对得上还不够，长相也要对得上：误读的号（102 读成 101）会把真小节并掉。
        target = next(
            (
                item
                for item in neighbours
                if item.number is not None
                and any(_same_or_extra(number, item.number) for number in heard)
                and loosely_same(item.latest, slot.latest)
            ),
            None,
        )
        if target is not None:
            target.seen += slot.seen
            room = MAX_OBSERVATIONS - len(target.observations)
            target.observations.extend(slot.observations[: max(0, room)])
            absorbed += 1
            continue
        # 读到的号全都已有别的槽位、长相也和其中之一对得上：这是追踪时把几个小节的观测混在一起的多余槽位。
        if (
            heard
            and all(number in owners for number in heard)
            and any(loosely_same(owners[number].latest, slot.latest) for number in heard)
        ):
            absorbed += 1
            continue
        result.append(slot)
    return result, absorbed


def _same_or_extra(read: int, number: int) -> bool:
    """读数就是这个号，或者只是在前后多读出一位（旁边的笔画被当成了数字）。"""
    text, want = str(read), str(number)
    return text == want or (len(text) == len(want) + 1 and (text.startswith(want) or text.endswith(want)))


def _truncated(read: int, number: int) -> bool:
    """读数是这个号截掉头或尾剩下的部分（17 只剩 1、157 只剩 57）。"""
    text, want = str(read), str(number)
    return len(text) < len(want) and (want.startswith(text) or want.endswith(text))


def _reorder(slots: list[Slot]) -> tuple[list[Slot], int, int]:
    """按号的最长递增子序列定骨架；不在骨架上的有号槽位，号没被占又有把握就挪到号该在的位置，否则当误读清掉号。

    没有号的槽位和它前面那个槽位绑成一块一起挪。
    """
    numbered = [index for index, slot in enumerate(slots) if slot.number is not None]
    keep = set(_longest_increasing([slots[index].number for index in numbered]))
    spine = {numbered[position] for position in keep}
    taken = {slots[index].number for index in spine}
    # 骨架上的合并休止占着它到下一个骨架号之间的所有号：落在这里面的号是误读（112 截成 12）。
    covered: list[tuple[int, int]] = []
    ordered_spine = sorted(spine)
    for index, following in zip(ordered_spine, ordered_spine[1:]):
        if is_multirest(slots[index].latest):
            covered.append((slots[index].number, slots[following].number))
    spine_times = sorted((slots[index].number, slots[index].first_seconds) for index in spine)
    blocks: list[tuple[float, list[Slot]]] = []
    moved = misread = 0
    for index, slot in enumerate(slots):
        number = slot.number
        if number is None:
            if blocks:
                blocks[-1][1].append(slot)
            else:
                blocks.append((float("-inf"), [slot]))
            continue
        if index in spine:
            blocks.append((float(number), [slot]))
            continue
        inside_rest = any(start < number < end for start, end in covered)
        # 一个小节不会比它后面的小节晚出现很久（谱是先露出来再播到）：晚得多的是截断的号（21 只剩 1）。
        following = next((seconds for value, seconds in spine_times if value > number), None)
        too_late = following is not None and slot.first_seconds > following + _LATE
        # 比骨架上最大的号还大出一截：没有后一小节可比时间，多半是误读（181 读成 811），不能挪到最后。
        beyond = following is None and bool(spine_times) and number > spine_times[-1][0] + _BEYOND
        too_late = too_late or beyond
        if number not in taken and slot.number_score >= CONFIDENT and not inside_rest and not too_late:
            blocks.append((float(number), [slot]))
            taken.add(number)
            moved += 1
            continue
        if number in taken and slot.number_score >= _SURE_MERGE and not too_late:
            # 号已经有了、这一份又读得很准：是追踪收重的一份，挪到同号的旁边等合并。
            blocks.append((number + 0.5, [slot]))
            moved += 1
            continue
        slot.number = None
        slot.number_score = 0.0
        misread += 1
        if blocks:
            blocks[-1][1].append(slot)
        else:
            blocks.append((float("-inf"), [slot]))
    blocks.sort(key=lambda block: block[0])
    return [slot for _key, members in blocks for slot in members], moved, misread


def _longest_increasing(values: list[int | None]) -> list[int]:
    """严格递增的最长子序列，返回下标。"""
    import bisect

    tails: list[int] = []
    tail_index: list[int] = []
    parent = [-1] * len(values)
    for index, value in enumerate(values):
        assert value is not None
        position = bisect.bisect_left(tails, value)
        if position == len(tails):
            tails.append(value)
            tail_index.append(index)
        else:
            tails[position] = value
            tail_index[position] = index
        parent[index] = tail_index[position - 1] if position > 0 else -1
    result: list[int] = []
    current = tail_index[-1] if tail_index else -1
    while current != -1:
        result.append(current)
        current = parent[current]
    return result[::-1]


def gaps(slots: list[Slot]) -> list[list[int]]:
    """相邻两个有号槽位之间跳过的号。多小节休止也会出现在这里，只供核对。"""
    numbers = [slot.number for slot in slots if slot.number is not None]
    return [[a, b] for a, b in zip(numbers, numbers[1:]) if b - a > 1]
