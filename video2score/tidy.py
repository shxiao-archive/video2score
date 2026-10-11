"""把投票后的小节图收拾干净：谱线整条重画、清掉背景残渣、抹掉视频里的小节号。

输入是墨迹图（0~1，1 是纯黑）和谱线几何；原地修改。
- 谱线：五线谱整条画；TAB 在压着数字的地方断开（线上下半个线距内有一段够高的墨）。
- 残渣：落在谱表上方 5 个、下方 3 个线距以外的块、从画面边缘截进来的谱表外的块、大块的浅灰块。
- 小节号：左右小节线旁边、谱表上方的小数字块（排版时按正确的号重印）。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 画出来的谱线、小节线有多黑。
_LINE = 0.9


@dataclass
class Geometry:
    """一张小节图里的谱线位置（绝对 y）和小节线位置。"""

    spacing: float
    groups: list[tuple[list[float], bool]]  # (这组谱线的 y, 是不是 TAB)
    left: int
    right: int

    @property
    def top(self) -> float:
        return self.groups[0][0][0]

    @property
    def bottom(self) -> float:
        return self.groups[-1][0][-1]


def geometry(staff_top: float, lines: tuple[float, ...], layout, kind: str, spacing: float, left: int, right: int) -> Geometry:
    """从小节记录（相对 staff_top 的谱线、layout、谱类）算出几何。

    哪组是 TAB：layout 里标了就用标记；没标时，五线谱+TAB 的第二组、纯 TAB 谱、不是 5 条线的组是 TAB。
    """
    groups: list[tuple[list[float], bool]] = []
    start = 0
    for position, item in enumerate(layout):
        count = int(item[0])
        ys = [staff_top + float(value) for value in lines[start : start + count]]
        start += count
        marked = item[2] if len(item) > 2 else None
        if marked is not None:
            is_tab = marked == "tab"
        else:
            is_tab = count != 5 or kind == "tab" or (kind == "staff+tab" and position > 0)
        groups.append((ys, is_tab))
    return Geometry(spacing, groups, left, right)


def tidy(ink: np.ndarray, geo: Geometry) -> np.ndarray:
    geo = _snap_lines(ink, geo)
    ink = _drop_residue(ink, geo)
    ink = _drop_faint(ink)
    # 哪些列压着 TAB 数字，要在抹旧线之前看：抹线会带走数字贴着线的笔画。
    occupied = {
        int(round(y)): _occupied(ink, int(round(y)), geo.left, geo.right, geo.spacing, _thickness(geo.spacing))
        for ys, is_tab in geo.groups
        if is_tab
        for y in ys
    }
    _erase_old_lines(ink, geo, occupied)
    # 抹小节号放在抹旧线之后：压在顶线上的号这时才和谱线分开。
    _erase_labels(ink, geo)
    _redraw_lines(ink, geo, occupied)
    return ink


def _snap_lines(ink: np.ndarray, geo: Geometry) -> Geometry:
    """谱线位置对到图里旧线真正所在的行：检测出的线位是小数，四舍五入可能差一行，
    重画的线就会挪一像素、压到贴着线的符头边上。旧线不够明显时保持原位。"""
    groups: list[tuple[list[float], bool]] = []
    solid = ink[:, geo.left : geo.right + 1] >= 0.35
    need = 0.3 * solid.shape[1]
    for ys, is_tab in geo.groups:
        snapped = []
        for y in ys:
            row = int(round(y))
            if not 1 <= row < ink.shape[0] - 1:
                snapped.append(y)
                continue
            counts = {r: int(solid[r].sum()) for r in (row - 1, row, row + 1)}
            best = max(counts, key=lambda r: (counts[r], r == row))
            snapped.append(float(best) if counts[best] >= need and counts[best] > counts[row] else y)
        groups.append((snapped, is_tab))
    return Geometry(geo.spacing, groups, geo.left, geo.right)


def _thickness(spacing: float) -> int:
    return max(1, int(round(spacing * 0.09)))


def clusters(solid: np.ndarray, gap: int = 1) -> tuple[int, np.ndarray, np.ndarray]:
    """连通块，但只隔 gap 像素的碎点算一块（断成一串点的空心符头算一整块）：在膨胀过的图上分块。
    返回（块数，每个像素的块号，每块的外框和面积）。面积只数 solid 里的像素。"""
    solid = solid.astype(np.uint8)
    size = 2 * gap + 1
    grown = cv2.dilate(solid, np.ones((size, size), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(grown, connectivity=8)
    labels = np.where(solid > 0, labels, 0)
    stats[:, cv2.CC_STAT_AREA] = np.bincount(labels.ravel(), minlength=count)
    return count, labels, stats


def _drop_faint(ink: np.ndarray) -> np.ndarray:
    """整块都不够黑（没有一个像素过 0.6）的淡灰碎屑：投票没投干净的背景。"""
    faint = (ink >= 0.12).astype(np.uint8)
    count, labels, _stats = clusters(faint)
    if count < 2:
        return ink
    darkest = np.zeros(count, np.float32)
    np.maximum.at(darkest, labels.ravel(), ink.ravel())
    cleaned = ink.copy()
    cleaned[(darkest < 0.6)[labels] & (labels > 0)] = 0.0
    return cleaned


def _erase_old_lines(ink: np.ndarray, geo: Geometry, occupied: dict[int, np.ndarray]) -> None:
    """重画之前先抹掉原来的谱线（谱线碎段、错开一两像素的旧线）。

    只抹线附近几行里、竖着很薄、又属于一整段长横线（超过 1 个线距）的墨。数字、字母的横笔短，
    符头、符杠、符干在竖直方向厚，都留着；TAB 上压着数字的列不动。
    """
    spacing = geo.spacing
    thickness = _thickness(spacing)
    reach = thickness + 2
    long = max(5, int(round(1.0 * spacing)))
    solid = (ink >= 0.35).astype(np.uint8)
    thin = solid & ~cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((thickness + 3, 1), np.uint8)).astype(bool)
    horizontal = cv2.morphologyEx(thin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, long), np.uint8)).astype(bool)
    for ys, is_tab in geo.groups:
        for y in ys:
            row = int(round(y))
            y0, y1 = max(0, row - reach), min(ink.shape[0], row + reach + 1)
            erase = horizontal[y0:y1, geo.left : geo.right + 1].copy()
            if is_tab and row in occupied:
                erase[:, occupied[row][: erase.shape[1]]] = False
            band = ink[y0:y1, geo.left : geo.right + 1]
            band[erase] = 0.0


def _redraw_lines(ink: np.ndarray, geo: Geometry, occupied: dict[int, np.ndarray]) -> None:
    """按检测到的线位画标准粗细的直线，TAB 在压着数字的列断开；并重画两侧小节线。"""
    spacing = geo.spacing
    thickness = _thickness(spacing)
    width = geo.right - geo.left + 1
    for ys, is_tab in geo.groups:
        for y in ys:
            row = int(round(y))
            y0 = max(0, row - thickness // 2)
            draw = np.ones(width, bool)
            if is_tab and row in occupied:
                draw = ~occupied[row]
            segment = ink[y0 : y0 + thickness, geo.left : geo.right + 1]
            segment[:, draw[: segment.shape[1]]] = np.maximum(segment[:, draw[: segment.shape[1]]], _LINE)
        y0 = int(round(ys[0]))
        y1 = int(round(ys[-1])) + 1
        for x in (geo.left, geo.right):
            x0 = max(0, x - thickness // 2)
            ink[y0:y1, x0 : x0 + thickness] = np.maximum(ink[y0:y1, x0 : x0 + thickness], _LINE)


def _occupied(ink: np.ndarray, row: int, left: int, right: int, spacing: float, thickness: int) -> np.ndarray:
    """TAB 线上压着数字的列：线上下半个线距内，这一列有好几行墨（数字的笔画）。

    符干只有一两像素宽、延音弧只有一两行高，都不算：线从它们下面画过去。
    """
    reach = int(round(0.5 * spacing))
    above = ink[max(0, row - reach) : max(0, row - thickness), left : right + 1] >= 0.5
    below = ink[row + thickness + 1 : row + reach + 1, left : right + 1] >= 0.5
    rows = above.sum(axis=0) + below.sum(axis=0)
    tall = (rows >= max(3, int(round(0.2 * spacing)))).astype(np.uint8)[None, :]
    # 太窄的（符干）不算；剩下的左右各让出两像素。
    tall = cv2.morphologyEx(tall, cv2.MORPH_OPEN, np.ones((1, max(2, int(round(0.2 * spacing)))), np.uint8))
    tall = cv2.dilate(tall, np.ones((1, 5), np.uint8))
    return tall[0] > 0


def _drop_residue(ink: np.ndarray, geo: Geometry) -> np.ndarray:
    """按连通块清背景残渣。"""
    spacing = geo.spacing
    solid = (ink >= 0.35).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(solid, connectivity=8)
    height, width = ink.shape
    # 谱表上方 3 个线距以内的都留（速度、滑音记号、连音数字）；更高处只留粗体的块（合并休止的小节数）。
    zone_top = geo.top - 3.0 * spacing
    bold_top = geo.top - 5.5 * spacing
    zone_bottom = geo.bottom + 3.0 * spacing
    staff_top = geo.top - 0.5 * spacing
    staff_bottom = geo.bottom + 0.5 * spacing
    # 只有合并休止的小节，谱表上方高处才可能有东西（它的小节数，粗体）。
    multirest = has_multirest(ink, geo)
    drop = np.zeros(count, bool)
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        y1 = y + h
        # 完全在谱表下方 3 个线距以外；上方 3 个线距以外的，除非是粗体。
        if y > zone_bottom or y1 < bold_top:
            drop[index] = True
            continue
        if y1 < zone_top and not (multirest and _is_bold(labels[y:y1, x : x + w] == index, spacing)):
            drop[index] = True
            continue
        outside = y1 < staff_top or y > staff_bottom
        # 谱表外、碰到画面边缘：从邻格截进来的残块、画面外伸进来的插画。
        if outside and (x <= 0 or x + w >= width or y <= 0 or y1 >= height):
            drop[index] = True
            continue
        # 大块的浅灰：没投掉的背景画。
        if area > 4 * spacing * spacing:
            part = labels[y:y1, x : x + w] == index
            if float(ink[y:y1, x : x + w][part].mean()) < 0.55:
                drop[index] = True
    cleaned = ink.copy()
    cleaned[drop[labels]] = 0.0
    return cleaned


def has_multirest(ink: np.ndarray, geo: Geometry) -> bool:
    """五线谱中线上一道粗横杠（合并休止）：中线上下 0.4 个线距里，有好几行墨连成小节宽度的四分之一以上。"""
    ys, _is_tab = geo.groups[0]
    middle = int(round((ys[0] + ys[-1]) / 2))
    reach = int(round(0.4 * geo.spacing))
    band = ink[max(0, middle - reach) : middle + reach + 1, geo.left : geo.right + 1] >= 0.5
    need = 0.25 * (geo.right - geo.left)
    long_rows = 0
    for row in band:
        padded = np.concatenate(([0], row.astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(padded))
        runs = edges[1::2] - edges[::2]
        if runs.size and runs.max() >= need:
            long_rows += 1
    # 谱线本身占一两行；粗横杠要比它厚得多。
    return long_rows >= max(4, int(round(0.35 * geo.spacing)))


def _is_bold(mask: np.ndarray, spacing: float) -> bool:
    """粗体的块：够高，腐蚀掉 0.15 个线距以后还剩三成以上的墨（细的文字、插画线条剩不了多少）。"""
    if mask.shape[0] < 0.8 * spacing:
        return False
    size = max(2, int(round(0.15 * spacing)))
    total = int(mask.sum())
    return total > 0 and cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8)).sum() >= 0.3 * total


def _erase_labels(ink: np.ndarray, geo: Geometry) -> None:
    """视频里印的小节号：小节线左 2.5 到右 5.5 个线距（右边线只看左侧，那是下一小节的号露进来），
    谱表顶线上方 3 个线距以内、不连谱线、个头不超过 1.1 个线距的块。"""
    spacing = geo.spacing
    top = geo.top
    y0 = max(0, int(top - 3.0 * spacing))
    # 号的底边可能压着顶线（抹掉旧线以后它和谱线分开了）。
    y1 = int(top + 0.3 * spacing)
    if y1 <= y0:
        return
    boxes = [
        (max(0, int(geo.left - 2.5 * spacing)), int(geo.left + 5.5 * spacing)),
        (max(0, int(geo.right - 2.5 * spacing)), ink.shape[1]),
    ]
    solid = (ink >= 0.35).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(solid, connectivity=8)
    frames = [
        (int(x), int(y), int(x + w), int(y + h))
        for x, y, w, h, _area in stats[1:]
        if h > 1.1 * spacing or w > 2.5 * spacing
    ]
    for index in range(1, count):
        x, y, w, h, _area = (int(value) for value in stats[index])
        if y < y0 - 2 or y + h > y1 or h > 1.1 * spacing or w > 2.5 * spacing:
            continue
        # 被别的块框住的字（排练号方框里的字母）不是小节号。
        if any(fx0 <= x and x + w <= fx1 and fy0 <= y and y + h <= fy1 for fx0, fy0, fx1, fy1 in frames):
            continue
        if any(a <= x and x + w <= b for a, b in boxes):
            region = labels[y : y + h, x : x + w] == index
            ink[y : y + h, x : x + w][region] = 0.0
