"""把一个槽位的多次观测合成一张干净的小节图：对齐、逐像素投票、去残点、重画谱线。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

from video2score.measures import PAD_ABOVE, Measure
from video2score.tidy import clusters, geometry, tidy
from video2score.track import Slot

# darkness 低于这个值当纸，高于 _FULL 当纯黑，中间线性过渡，保留一点抗锯齿。
_PAPER = 70.0
_FULL = 170.0


@dataclass
class CleanMeasure:
    """一张干净的小节图。image 是白底黑字的灰度图。坐标都以 image 左上角为原点。"""

    image: np.ndarray
    spacing: float
    staff_top: float
    lines: tuple[float, ...]
    layout: tuple[tuple[int, float], ...]
    kind: str
    left_bar: int
    right_bar: int
    # 参考观测那一帧的灰度图（白底黑字，没投票、没压成黑白）。识别读这一张：细笔画在干净图里会被压淡或投掉。
    raw: np.ndarray | None = None
    # 另外几帧（和参考帧对齐过、看得到整个小节、时间上错开）。识别时每帧各读一遍再投票，单帧的缺笔、背景干扰能互相补上。
    extra_raws: list[np.ndarray] = field(default_factory=list)


# 除参考帧外，最多再留几帧给识别投票。
EXTRA_RAWS = 4


def render_slot(slot: Slot) -> CleanMeasure:
    observations = _same_width(slot)
    # 读到了本格号的观测是锚：槽位里混进了长得差不多的别的小节时，按锚选，不按多数选。
    anchors = [item for item in observations if slot.number is not None and item.number == slot.number]
    reference = max(anchors or observations, key=lambda item: item.valid[1] - item.valid[0])
    stack, seen, used, tinted = _aligned_stack(reference, observations)
    # 变了色的墨（正在播放的音符、光标）这一帧不算看到，交给别的帧投票；每帧都变色的地方最后再补。
    valid = seen & (tinted == 0)
    # 槽位里偶尔混进别的小节的观测（翻页、淡入淡出时收错）：只留和锚（没有锚时和大多数）长得一样的。
    keep = coherent(stack, valid, [index for index, item in enumerate(used) if any(item is anchor for anchor in anchors)])
    stack, valid, seen, tinted = stack[keep], valid[keep], seen[keep], tinted[keep]
    used = [used[index] for index in keep]
    everything = _vote(stack, valid)
    # 只让背景最干净的几帧投票：整片发白、光标、深色背景的帧会把符头、谱线挖空或染黑。
    ranked = _cleanliness(stack, valid, everything)
    keep = ranked[: max(3, min(5, len(ranked) // 2))] if len(ranked) >= 3 else list(range(len(stack)))
    darkness = np.minimum(_vote(stack[keep], valid[keep]), _floor(stack[keep], valid[keep]))
    # 最干净的几帧里这个像素都是彩色的（音符正在播放），用全部帧的投票。
    darkness = np.where(valid[keep].any(axis=0), darkness, everything)
    darkness = _restore_thin(darkness, stack[keep], valid[keep])
    darkness = _fill_tinted(darkness, valid, seen, tinted)
    ink = np.clip((darkness - _PAPER) / (_FULL - _PAPER), 0.0, 1.0)
    ink = _drop_specks(ink, reference.spacing)
    ink = _drop_gray_blobs(ink, reference.spacing)
    staff_top = PAD_ABOVE * reference.spacing
    margin = reference.left - reference.origin[0]
    right_bar = margin + reference.width
    geo = geometry(staff_top, reference.lines, reference.layout, reference.kind, reference.spacing, margin, right_bar)
    ink = tidy(ink, geo)
    image = (255.0 - ink * 255.0).round().astype(np.uint8)
    return CleanMeasure(
        image=image,
        spacing=reference.spacing,
        staff_top=staff_top,
        lines=reference.lines,
        layout=reference.layout,
        kind=reference.kind,
        left_bar=margin,
        right_bar=right_bar,
        # 识别用单独一帧：多帧投票会把一两像素宽的细笔画投掉（各帧错开一个像素就对不上）。
        **_raws(reference, used, stack, valid, everything),
    )


def render_prefix(slots: list[Slot]) -> np.ndarray | None:
    """乐谱最开头、第一小节左边那一段（竖排的乐器名、TAB 左边的调弦字母）：最前面几格里第一个截到的，各次观测投票。

    返回白底黑字的灰度图，行和那一格的小节图对齐（谱表顶线在 PAD_ABOVE 个线距处）。没有墨就返回 None。
    """
    for slot in slots[:3]:
        patches = [item.prefix for item in slot.observations if item.prefix is not None]
        patches = [patch for patch in patches if patch.shape == patches[0].shape] if patches else []
        if not patches:
            continue
        spacing = slot.latest.spacing
        darkness = np.median(np.stack(patches).astype(np.float32), axis=0)
        ink = np.clip((darkness - _PAPER) / (_FULL - _PAPER), 0.0, 1.0)
        ink = _drop_gray_blobs(_drop_specks(ink, spacing), spacing)
        # 只留挨着谱表（上 1.5、下 0.7 个线距以内）、不碰左边缘（截进来的插画）的块：邻格的号、插画碎块不要。
        top = PAD_ABOVE * spacing - 1.5 * spacing
        bottom = PAD_ABOVE * spacing + (slot.latest.lines[-1] if slot.latest.lines else 0) + 0.7 * spacing
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats((ink >= 0.35).astype(np.uint8), connectivity=8)
        drop = np.zeros(count, bool)
        for index in range(1, count):
            x, y, w, h, _area = (int(value) for value in stats[index])
            drop[index] = x <= 0 or y + h < top or y > bottom
        ink[drop[labels]] = 0.0
        if float((ink >= 0.5).sum()) < spacing * spacing:
            return None
        # 左边的空白裁掉，右边贴着第一小节，保持原来的间距。
        columns = np.flatnonzero((ink >= 0.35).any(axis=0))
        ink = ink[:, max(0, int(columns[0] - 0.5 * spacing)) :]
        return (255.0 - ink * 255.0).round().astype(np.uint8)
    return None


def _same_width(slot: Slot) -> list[Measure]:
    """同一个小节不管什么时候再出现，宽度都一样（小节线只差一两像素）。宽度明显不同的观测是按截断的号
    并进来的别的小节（157 读成 57）：按宽度分组，留读到本格号最多、观测最多的那一组。"""
    groups: list[list[Measure]] = []
    for item in slot.observations:
        tol = max(4.0, 0.015 * item.width)
        group = next((members for members in groups if abs(members[0].width - item.width) <= tol), None)
        if group is None:
            groups.append([item])
        else:
            group.append(item)
    if len(groups) == 1:
        return slot.observations

    def weight(members: list[Measure]) -> tuple[int, int]:
        return sum(1 for item in members if slot.number is not None and item.number == slot.number), len(members)

    return max(groups, key=weight)


# 两次观测的笔画重合度（Dice）低于这个就不是同一个小节。
_SAME_INK = 0.8


def coherent(stack: np.ndarray, valid: np.ndarray, anchors: list[int] | None = None) -> list[int]:
    """和"最像其他所有观测的那一次"（中心）笔画重合度够高的观测。给了锚时，中心只从锚里挑。

    只比细的深色笔画（音符、数字、谱线）：背景画每帧都不一样，比整张图、比大片深色都分不出是不是同一个小节。
    """
    count = len(stack)
    if count <= 2 and not anchors:
        return list(range(count))
    solid = np.stack([_strokes(layer) for layer in stack])
    grown = np.stack([cv2.dilate(layer.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for layer in solid])
    similarity = np.eye(count)
    for i in range(count):
        for j in range(i + 1, count):
            mask = valid[i] & valid[j]
            a, b = solid[i] & mask, solid[j] & mask
            total = int(a.sum() + b.sum())
            value = float((a & grown[j]).sum() + (b & grown[i]).sum()) / total if total else 1.0
            similarity[i, j] = similarity[j, i] = value
    candidates = anchors or list(range(count))
    centre = max(candidates, key=lambda index: float(similarity[index].sum()))
    return [index for index in range(count) if similarity[centre, index] >= _SAME_INK]


def _strokes(darkness: np.ndarray) -> np.ndarray:
    """细的深色笔画：黑顶帽去掉大片的深色背景（比 9 像素宽的色块），只留音符、数字、谱线。"""
    image = np.clip(darkness, 0, 255).astype(np.uint8)
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.subtract(image, background) >= 90


def _cleanliness(stack: np.ndarray, valid: np.ndarray, voted: np.ndarray) -> list[int]:
    """各帧按干净程度排序：投票结果里是纸的地方这一帧有多黑，加上是墨的地方这一帧缺了多少。"""
    paper = voted < _PAPER
    ink = voted >= _FULL
    scored: list[tuple[float, int]] = []
    for index in range(len(stack)):
        mask = valid[index]
        on_paper, on_ink = paper & mask, ink & mask
        if not on_paper.any():
            continue
        noise = float(np.mean(stack[index][on_paper]))
        missing = float(np.mean(np.clip(voted[on_ink] - stack[index][on_ink], 0, None))) if on_ink.any() else 0.0
        scored.append((noise + missing, index))
    return [index for _score, index in sorted(scored)]


def _raws(
    reference: Measure,
    used: list[Measure],
    stack: np.ndarray,
    valid: np.ndarray,
    voted: np.ndarray,
) -> dict[str, object]:
    """给识别用的单帧图：看得到整个小节的观测里，背景最干净的几帧。

    干净的标准：投票结果里是纸的地方，这一帧有多黑（背景画、光标）；是墨的地方，这一帧缺了多少（被挡住）。
    两样加起来越小越干净。最干净的一帧当主帧（raw），其余的留给识别投票（extra_raws）。都已经按参考帧对齐。

    "看得到整个小节"只看识别用得着的那一段：五线谱第一线往上 3 个线距到底。更上面是小节号、速度记号，
    谱面框的上沿在帧与帧之间会跳几十像素（有的视频一半的帧从第 0 行看起、一半从第 41 行），按整张图比会把一半的帧排除掉。
    """
    top = max(reference.valid[0], int(PAD_ABOVE * reference.spacing - 3.0 * reference.spacing))
    needed = np.zeros(stack.shape[1], bool)
    needed[top : reference.valid[1]] = True
    rows = valid.any(axis=2)
    ranked = [index for index in _cleanliness(stack, valid, voted) if (rows[index] & needed).sum() >= 0.95 * needed.sum()]
    if not ranked:
        return {"raw": (255 - reference.darkness).astype(np.uint8), "extra_raws": []}
    picks = ranked[: 1 + EXTRA_RAWS]
    images = [(255 - np.clip(stack[index], 0, 255)).astype(np.uint8) for index in picks]
    return {"raw": images[0], "extra_raws": images[1:]}


def _aligned_stack(
    reference: Measure,
    observations: list[Measure],
) -> tuple[np.ndarray, np.ndarray, list[Measure], np.ndarray]:
    """各次观测按参考图对齐（上下左右各挪几像素），叠成一摞。valid 标出每张里真正看到的行，used 是用上的观测，
    最后一项是同样对齐的彩色像素深浅（Measure.tinted）。"""
    height, width = reference.darkness.shape
    layers: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    tints: list[np.ndarray] = []
    used: list[Measure] = []
    for item in observations:
        patch, rows = item.darkness, item.valid
        dx, dy = (0, 0) if item is reference else _best_shift(reference, item)
        layer = np.zeros((height, width), np.float32)
        mask = np.zeros((height, width), bool)
        src = patch.astype(np.float32)
        # 目标 [y, x] 取自源 [y - dy, x - dx]，超出源图的部分不算看到。
        ys0, ys1 = max(0, dy), min(height, patch.shape[0] + dy)
        xs0, xs1 = max(0, dx), min(width, patch.shape[1] + dx)
        if ys1 <= ys0 or xs1 <= xs0:
            continue
        layer[ys0:ys1, xs0:xs1] = src[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
        tint = np.zeros((height, width), np.float32)
        if item.tinted is not None:
            tint[ys0:ys1, xs0:xs1] = item.tinted[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
        lo = max(ys0, rows[0] + dy)
        hi = min(ys1, rows[1] + dy)
        if hi > lo:
            mask[lo:hi, xs0:xs1] = True
        layers.append(layer)
        masks.append(mask)
        tints.append(tint)
        used.append(item)
    return np.stack(layers), np.stack(masks), used, np.stack(tints)


def _best_shift(reference: Measure, item: Measure) -> tuple[int, int]:
    """小节线和谱线已经大致对齐，只在 ±3 像素里找墨迹最重合的位置。"""
    base = _plain(reference)
    other = _plain(item)
    rows = (max(reference.valid[0], item.valid[0]) + 3, min(reference.valid[1], item.valid[1]) - 3)
    if rows[1] - rows[0] < 8:
        return 0, 0
    height = min(base.shape[0], other.shape[0])
    width = min(base.shape[1], other.shape[1])
    best = (float("inf"), 0, 0)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            y0, y1 = rows[0], min(rows[1], height)
            a = base[y0:y1, 3 : width - 3]
            b = other[y0 - dy : y1 - dy, 3 - dx : width - 3 - dx]
            if a.shape != b.shape or a.size == 0:
                continue
            score = float(np.mean(np.abs(a - b)))
            if score < best[0]:
                best = (score, dx, dy)
    return best[1], best[2]


def _plain(measure: Measure) -> np.ndarray:
    """不管颜色的深浅：变色的音符也算墨。对齐用它，变色的帧才对得准。"""
    darkness = measure.darkness.astype(np.float32)
    return darkness if measure.tinted is None else np.maximum(darkness, measure.tinted.astype(np.float32))


def _fill_tinted(voted: np.ndarray, valid: np.ndarray, seen: np.ndarray, tinted: np.ndarray) -> np.ndarray:
    """每一帧里都是彩色的像素（整个槽位都在播放时才看到的音符）没有帧可投，用彩色像素本来深浅的中位数。"""
    lost = seen.any(axis=0) & ~valid.any(axis=0)
    if not lost.any():
        return voted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(seen, tinted, np.nan), axis=0)
    return np.where(lost, np.nan_to_num(median, nan=0.0), voted)


def _vote(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """逐像素投票。三次以上取中位数；两次取较浅的；一次就用它。

    谱带后面的插画在不同帧里不一样，投票会把它去掉；音符每次都在，留下来。
    """
    count = valid.sum(axis=0)
    masked = np.where(valid, stack, np.nan)
    # 一次都没看到的像素（被画面边缘截掉）全是 NaN，结果记为纸色。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(masked, axis=0)
        lighter = np.nanmin(masked, axis=0)
    result = np.where(count >= 3, median, lighter)
    return np.nan_to_num(result, nan=0.0)


def _floor(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """墨要在几乎每一帧里都有（左右上下一像素内）：取各帧（先膨胀一像素）的第二浅值，容一帧例外。

    背景画随谱带移动，在最干净的几帧里也可能碰巧有三帧落在同一处，中位数去不掉；
    音符每一帧都在。没看到的帧不算，只剩一两帧看到时取最浅的。
    """
    grown = np.stack([cv2.dilate(layer, np.ones((3, 3), np.uint8)) for layer in stack])
    masked = np.where(valid, grown, np.inf)
    ordered = np.sort(masked, axis=0)
    count = valid.sum(axis=0)
    pick = np.where(count >= 4, 1, 0)
    floor = np.take_along_axis(ordered, pick[None], axis=0)[0]
    return np.where(np.isfinite(floor), floor, 0.0).astype(np.float32)


def _restore_thin(voted: np.ndarray, stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """补回中位数投掉的细笔画：一两像素宽的符干、空心符头的边，各帧错开一像素就对不上，中位数会把它投掉。

    最干净那帧（stack[0]）的笔画，在一半以上的帧里左右一像素之内也有，就是真的笔画，补回来。
    """
    count = len(stack)
    if count < 3:
        return voted
    kernel = np.ones((1, 3), np.uint8)
    marks = [(stack[index] >= _PAPER) & valid[index] for index in range(count)]
    support = sum(cv2.dilate(mark.astype(np.uint8), kernel) for mark in marks[1:])
    real = marks[0] & (support * 2 >= count - 1)
    return np.where(real, np.maximum(voted, stack[0]), voted)


def _drop_gray_blobs(ink: np.ndarray, spacing: float) -> np.ndarray:
    """谱表里没投掉的灰色背景块：离纯黑笔画两像素以外、又有三像素以上厚、面积不小的灰色一片。

    音符、数字的芯都是纯黑，灰的只是它们的边；细的灰线（一两像素的符干、空心符头的边）没有这么厚，留着。
    背景块常和谱线连成一片，按连通块删不掉，这里只看灰色部分本身。
    """
    core = cv2.dilate((ink >= 0.85).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gray = ((ink >= 0.2) & ~core).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)
    if count < 2:
        return ink
    thick = cv2.erode(gray, np.ones((3, 3), np.uint8))
    thick_count = np.bincount(labels[thick > 0], minlength=count)
    blob = (stats[:, cv2.CC_STAT_AREA] >= (0.5 * spacing) ** 2) & (thick_count >= 10)
    blob[0] = False
    cleaned = ink.copy()
    cleaned[blob[labels]] = 0.0
    return cleaned


def _drop_specks(ink: np.ndarray, spacing: float) -> np.ndarray:
    """去掉孤立的小点：压缩噪点、没投掉的插画碎屑。比附点小很多的才去。"""
    solid = (ink >= 0.5).astype(np.uint8)
    # 空心符头、细弧的抗锯齿边常断成隔一两像素的一串点，连成一簇再看大小。
    count, labels, stats = clusters(solid, gap=2)
    limit = max(4, int(round((spacing * 0.18) ** 2)))
    small = np.zeros(count, bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < limit:
            small[label] = True
    cleaned = ink.copy()
    cleaned[small[labels]] = 0.0
    # 投票后残留的淡灰雾（不到半黑、也不贴着任何笔画）一起抹掉。
    kept = ((solid > 0) & ~small[labels]).astype(np.uint8)
    near = cv2.dilate(kept, np.ones((5, 5), np.uint8)) > 0
    cleaned[(cleaned < 0.5) & ~near] = 0.0
    return cleaned
