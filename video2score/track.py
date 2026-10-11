"""跨帧追踪小节。同一个小节在很多帧里出现，只占一个槽位。

判断"是不是同一个小节"不只看长相：贝斯谱里一连八小节 `4 4 4 4` 很常见，
按长相去重会把它们并成一个。这里先求相邻两帧之间整体移了多少（横滚、下滚或不动），
再按位置对应。长相一样、位置对不上的，是新的小节。

取帧要够密，让相邻两帧之间的位移小于半个小节。这样多个位移都能对上时，取最小的那个就对。
翻页或原地换段时位移对不上，再拿本帧开头和已收序列的末尾比，接上重叠的部分。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from video2score.measures import (
    FrameMeasures,
    Measure,
    labels_agree,
    loosely_same,
    same_measure,
)
from video2score.ocr import read_measure

# 每个槽位最多留几次观测。横滚时尽量留下不同位置的，背景不同，投票才去得掉插画。
MAX_OBSERVATIONS = 6
# 翻页时和已收序列末尾比较的范围。
_TAIL = 48
# 对不上时往前找几帧。
_HISTORY = 8
# 小节号读得这么有把握时，号不同就一定不是同一个小节。
_SURE = 0.75


@dataclass
class Slot:
    id: int
    observations: list[Measure] = field(default_factory=list)
    seen: int = 0
    first_seconds: float = 0.0
    # 读出的小节号和把握（0~1）。order.arrange 填写。
    number: int | None = None
    number_score: float = 0.0

    @property
    def latest(self) -> Measure:
        return self.observations[-1]

    def observe(self, measure: Measure) -> None:
        if self.seen == 0:
            self.first_seconds = measure.seconds
        self.seen += 1
        kept = self.observations
        if len(kept) < 2:
            kept.append(measure)
            return
        # 位置和时间都和之前差不多时，只替换最后一次，保证 latest 总是最新的。
        # 原地不动的小节（整页翻的谱）背景动画也在变，隔一阵的帧一样有用。
        moved = min(_apart(measure, item) for item in kept[:-1])
        if moved >= max(12.0, measure.spacing):
            if len(kept) < MAX_OBSERVATIONS:
                kept.append(measure)
                return
            # 满了就挤掉和别人最近的那一个，留下的位置、时间尽量分散。
            crowded = min(
                range(len(kept) - 1),
                key=lambda index: min(
                    _apart(kept[index], other) for position, other in enumerate(kept) if position != index
                ),
            )
            kept.pop(crowded)
            kept.append(measure)
            return
        kept[-1] = measure


# 时间差折算成位移：隔一秒算挪了这么多像素。
_PIXELS_PER_SECOND = 20.0


def _apart(a: Measure, b: Measure) -> float:
    """两次观测差多远：位置差加上折算的时间差。"""
    return abs(a.left - b.left) + abs(a.top - b.top) + _PIXELS_PER_SECOND * abs(a.seconds - b.seconds)


@dataclass
class Transition:
    """相邻两帧之间怎么对上的。kind 是 motion、tail 或 new。"""

    seconds: float
    kind: str
    offset: tuple[float, float] = (0.0, 0.0)
    matched: int = 0
    added: int = 0


class Tracker:
    def __init__(self) -> None:
        self.slots: list[Slot] = []
        self.transitions: list[Transition] = []
        # 最近几帧。和上一帧对不上时（转场、闪过一帧放大画面），再往前找。
        self._history: list[tuple[FrameMeasures, list[Slot]]] = []
        self._spacings: list[float] = []
        self._next_id = 0

    def add(self, current: FrameMeasures) -> list[Slot]:
        """收下一帧，返回这一帧每个小节对应的槽位（按阅读顺序）。"""
        if not current.measures:
            return []
        if self._zoomed(current):
            self.transitions.append(Transition(current.frame.seconds, "zoom"))
            return []
        measures = current.measures
        for measure in measures:
            reading = read_measure(measure)
            if reading is not None:
                measure.number, measure.number_score = reading.number, reading.score
        assigned: list[Slot | None] = [None] * len(measures)
        kind = "new"
        offset = (0.0, 0.0)
        for previous, previous_slots in reversed(self._history):
            motion = _match_motion(previous, previous_slots, current)
            if motion is not None:
                offset, assigned = motion
                kind = "motion"
                break
        if kind == "new" and self.slots:
            tail = self._match_tail(measures)
            if tail is not None:
                assigned = tail
                kind = "tail"
        if kind == "motion" and abs(offset[0]) + abs(offset[1]) < 1.0:
            self._match_earlier(measures, assigned)
        matched = sum(1 for slot in assigned if slot is not None)
        added = self._place(measures, assigned)
        self.transitions.append(
            Transition(current.frame.seconds, kind, offset, matched, added)
        )
        final = [slot for slot in assigned if slot is not None]
        for measure, slot in zip(measures, final):
            slot.observe(measure)
        self._history.append((current, final))
        del self._history[:-_HISTORY]
        return final

    def _zoomed(self, current: FrameMeasures) -> bool:
        """线距和前面多数帧差得多：缩放动画、画中画，整帧不参与追踪。
        只是其中几个小节差得多（背景里一道道等距的横条也被当成了谱行）时，丢掉那几个，其余照常追踪。"""
        recent = self._spacings[-60:]
        if len(recent) < 5:
            self._spacings.append(current.spacing)
            return False
        typical = float(np.median(recent))
        kept = [measure for measure in current.measures if abs(measure.spacing - typical) <= 0.12 * typical]
        if not kept:
            self._spacings.append(current.spacing)
            return True
        current.measures = kept
        self._spacings.append(float(np.median([measure.spacing for measure in kept])))
        return False

    def _place(self, measures: list[Measure], assigned: list[Slot | None]) -> int:
        """没对上的小节开新槽位，插在本帧里它前面那个已知小节之后。"""
        added = 0
        anchor: Slot | None = None
        pending: list[int] = []
        for index, slot in enumerate(assigned):
            if slot is not None:
                if pending:
                    # 本帧开头就出现的新小节（往回滚时）放在第一个已知小节之前。
                    position = self.slots.index(slot) if anchor is None else self.slots.index(anchor) + 1
                    for offset, item in enumerate(pending):
                        new = self._new_slot()
                        self.slots.insert(position + offset, new)
                        assigned[item] = new
                        added += 1
                    pending = []
                anchor = assigned[index]
                continue
            pending.append(index)
        if pending:
            position = len(self.slots) if anchor is None else self.slots.index(anchor) + 1
            for offset, item in enumerate(pending):
                new = self._new_slot()
                self.slots.insert(position + offset, new)
                assigned[item] = new
                added += 1
        return added

    def _match_earlier(self, measures: list[Measure], assigned: list[Slot | None]) -> None:
        """谱面没动，却有小节对不上上一帧：上一帧里它被播放光标切成了两半、或者被遮住了。
        往前几帧找同一位置、长相一样的小节，接到它的槽位上。"""
        taken = {id(slot) for slot in assigned if slot is not None}
        for index, measure in enumerate(measures):
            if assigned[index] is not None:
                continue
            tol = max(6.0, 0.4 * measure.spacing)
            for previous, previous_slots in reversed(self._history):
                found = next(
                    (
                        slot
                        for old, slot in zip(previous.measures, previous_slots)
                        if id(slot) not in taken
                        and abs(old.left - measure.left) <= tol
                        and abs(old.right - measure.right) <= tol
                        and abs(old.top - measure.top) <= tol
                        and loosely_same(old, measure)
                        and not numbers_conflict(old, measure)
                    ),
                    None,
                )
                if found is not None:
                    assigned[index] = found
                    taken.add(id(found))
                    break

    def _new_slot(self) -> Slot:
        slot = Slot(self._next_id)
        self._next_id += 1
        return slot

    def _match_tail(self, measures: list[Measure]) -> list[Slot | None] | None:
        """翻页后，本帧开头若接着已收序列的末尾，重叠的部分对上已有槽位。

        至少要连着对上两个小节。只对上一个时，多半是新页恰好以一样的小节开头。
        """
        tail = self.slots[-_TAIL:]
        best: tuple[int, int] | None = None
        for start in range(len(tail)):
            span = min(len(measures), len(tail) - start)
            if span < 2:
                continue
            if all(
                same_measure(tail[start + k].latest, measures[k])
                and labels_agree(tail[start + k].latest, measures[k])
                and not numbers_conflict(tail[start + k].latest, measures[k])
                for k in range(span)
            ):
                if best is None or span > best[1]:
                    best = (start, span)
        if best is None and len(measures) == 1:
            # 整帧只有一个小节（片头、淡入淡出以后）：和末尾某个槽位位置一样、长相一样，就是它又出现了。
            only = measures[0]
            tol = max(6.0, 0.4 * only.spacing)
            for start in range(len(tail) - 1, -1, -1):
                seen = tail[start].latest
                if (
                    abs(seen.left - only.left) <= tol
                    and abs(seen.right - only.right) <= tol
                    and abs(seen.top - only.top) <= tol
                    and same_measure(seen, only)
                    and labels_agree(seen, only)
                    and not numbers_conflict(seen, only)
                ):
                    best = (start, 1)
                    break
        if best is None:
            return None
        start, span = best
        assigned: list[Slot | None] = [None] * len(measures)
        for k in range(span):
            assigned[k] = tail[start + k]
        return assigned


def _match_motion(
    previous: FrameMeasures,
    previous_slots: list[Slot],
    current: FrameMeasures,
) -> tuple[tuple[float, float], list[Slot | None]] | None:
    """求两帧之间的整体位移，按位移把本帧小节对到上一帧的槽位。对不上返回 None。"""
    old = previous.measures
    new = current.measures
    spacing = current.spacing
    tol_x = max(6.0, 0.4 * spacing)
    tol_y = max(4.0, 0.5 * spacing)
    candidates: list[tuple[float, float]] = []
    for a in old:
        for b in new:
            if same_measure(a, b):
                candidates.append((float(b.left - a.left), b.top - a.top))
    if not candidates:
        return None
    # 相近的位移只算一次。
    offsets: list[tuple[float, float]] = []
    for candidate in sorted(candidates, key=lambda item: abs(item[0]) + abs(item[1])):
        if all(abs(candidate[0] - o[0]) > tol_x or abs(candidate[1] - o[1]) > tol_y for o in offsets):
            offsets.append(candidate)
    best: tuple[tuple[bool, int, int, float], tuple[float, float]] | None = None
    for dx, dy in offsets:
        support = 0
        conflicts = 0
        expected = 0
        used: set[int] = set()
        for b in new:
            left = b.left - dx
            top = b.top - dy
            if not _was_visible(previous, left, left + b.width, top, tol_y):
                continue
            expected += 1
            for position, a in enumerate(old):
                if position in used:
                    continue
                if (
                    abs(a.left - left) <= tol_x
                    and abs(a.top - top) <= tol_y
                    and same_measure(a, b)
                    and not numbers_conflict(a, b)
                ):
                    used.add(position)
                    support += 1
                    if _numbers_differ(a, b) or (not labels_agree(a, b) and not _numbers_same(a, b)):
                        conflicts += 1
                    break
        if support == 0 or support < 0.6 * expected:
            continue
        # 往回走的位移只靠一个长得一样的小节撑着：反复段里两种小节交替出现（A B A B），翻页时很容易碰巧对上一个，不算。
        forward = dx <= tol_x and dy <= tol_y
        if not forward and support < 2:
            continue
        # 音符都对得上、小节号却大半对不上：是换成了一段长得一样的新小节，不是没动。
        # 挪动了的位移，正好一半对不上也不要：整页翻到一段一样的反复（比如 100~101 和 102~103 一模一样）时，
        # 新页常正好能挪一下对上两个小节，号只读准了一个。
        still = abs(dx) <= tol_x and abs(dy) <= tol_y
        if conflicts * 2 > support or (conflicts and conflicts * 2 == support and not still):
            continue
        # 谱是往左、往上走的。往回走的位移只在没有别的解释时才用。
        # 然后对上的越多越好，小节号冲突的越少越好（音符一样的相邻小节靠它分开）；
        # 再一样时位移越小越好（取帧够密时，真位移不到半个小节）。
        forward = dx <= tol_x and dy <= tol_y
        rank = (forward, support - conflicts, -conflicts, -(abs(dx) + abs(dy)))
        if best is None or rank > best[0]:
            best = (rank, (dx, dy))
    if best is None:
        return None
    dx, dy = best[1]
    # 位移定了以后，落在同一位置的小节放宽来认：背景动画、音符弹跳、
    # 小节线差几像素都会让长相差一点，但位置和整体都对得上，就是同一个。
    assigned: list[Slot | None] = [None] * len(new)
    used = set()
    for index, b in enumerate(new):
        left = b.left - dx
        right = b.right - dx
        top = b.top - dy
        for position, a in enumerate(old):
            if position in used:
                continue
            if abs(a.top - top) > tol_y:
                continue
            # 一边的小节线偶尔差几像素。左右区间大部分重合就算同一位置。
            overlap = min(a.right, right) - max(a.left, left)
            union = max(a.right, right) - min(a.left, left)
            if union <= 0 or overlap / union < 0.85:
                continue
            if loosely_same(a, b) and not numbers_conflict(a, b):
                assigned[index] = previous_slots[position]
                used.add(position)
                break
    return (dx, dy), assigned


def numbers_conflict(left: Measure, right: Measure) -> bool:
    """两边都清楚地读到了小节号，而且号不同：一定不是同一个小节。"""
    if left.number is None or right.number is None or left.number == right.number:
        return False
    return min(left.number_score, right.number_score) >= _SURE


def _numbers_differ(left: Measure, right: Measure) -> bool:
    """两边的号都读得还算有把握（不到"一定不是同一个"那么准）、又不一样：算一处冲突。
    反复段里一模一样的小节翻页后正好落在原来的位置上，小节号是唯一能分开它们的东西。
    门槛 0.7：0.6 时两个一样的相邻小节的号读得含糊，被当成冲突，同一小节的位移没对上，丢了一个。"""
    if left.number is None or right.number is None or left.number == right.number:
        return False
    return min(left.number_score, right.number_score) >= 0.7


def _numbers_same(left: Measure, right: Measure) -> bool:
    """两边的号都读出来了（0.6 以上）、而且一样：小节号那一块的图对不上（背景插画透过来、帧帧在变）也不算冲突。
    两次碰巧读成同一个错号的很少，门槛比 _numbers_differ 低。"""
    if left.number is None or left.number != right.number:
        return False
    return min(left.number_score, right.number_score) >= 0.6


def _was_visible(previous: FrameMeasures, left: float, right: float, top: float, tol_y: float) -> bool:
    """按这个位移倒推回去，上一帧里这个小节应当完整可见吗。

    小节线贴着画面边缘时检测不到，那个小节在上一帧里本来就不会被收，要留出一个线距。
    """
    x0, _y0, x1, _y1 = previous.box
    margin = previous.spacing
    if left < x0 + margin or right > x1 - margin:
        return False
    return any(abs(row - top) <= tol_y for row in previous.rows)


def ordered_slots(tracker: Tracker, min_seen: int = 3) -> list[Slot]:
    """最终的小节序列。

    只出现过一两帧的多半是翻页、转场时的残影；线距和全曲差很多的，是缩放动画里的画面。都不收。
    """
    kept = [slot for slot in tracker.slots if slot.seen >= min_seen]
    if not kept:
        return kept
    typical = float(np.median([slot.latest.spacing for slot in kept]))
    # 追踪收重的小节在 order.arrange 里按读出的小节号合并。
    return [slot for slot in kept if abs(slot.latest.spacing - typical) <= 0.12 * typical]


def fragments(tracker: Tracker, min_seen: int = 3) -> list[Slot]:
    """只出现过一两帧的槽位（ordered_slots 不收的那些），线距和全曲一致的。留给 order.rescue 补漏。"""
    kept = [slot for slot in tracker.slots if slot.seen >= min_seen]
    if not kept:
        return []
    typical = float(np.median([slot.latest.spacing for slot in kept]))
    return [
        slot
        for slot in tracker.slots
        if slot.seen < min_seen and abs(slot.latest.spacing - typical) <= 0.12 * typical
    ]


def summary(tracker: Tracker) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for transition in tracker.transitions:
        kinds[transition.kind] = kinds.get(transition.kind, 0) + 1
    return kinds
