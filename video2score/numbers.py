"""读小节左上角的小节号。数字模板和各种残字规则都是按真实视频调出来的。"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

from video2score.staff import legacy_systems, without_staff_lines


# 左上角小节号的常见字形。排练号和品位数字不在这组里。
_DIGIT_TEMPLATES = (
    (9, "000000000000003399900000089900099100190000011900190000011900980000000950970000000790970000000790790000000690099900033990099900033990008899600950000000033900000000099800000000999000000008900000000008900000002299000000999900000000000000000000"),
    (0, "000000000000000089900000009900099000094400022900094400022900190000000900790000000940990000000990990000000990990000000990990000000990990000000990990000000980690000000940090000000900094400011900094400011900009900099000000099900000000000000000"),
    (6, "000000000000000000199990000009911000000099000000000099000000009990000000099900000000095500000000890079999000993300099900993300099900970000000980980000000990980000000790390000000890091100000910091100000910029900099900000099911000000000000000"),
    (8, "000000000000003699966000099600088900190000100910190000100910390000000940198500000900099750099100009999900000000399900000000399900000009604999100094400099910391100000980590000000880292200000840292200000840099600044910005799999000000000000000"),
    (1, "000000000000000459900000001559900000001119900000000009900000000009900000000009900000000009900000000009900000000009900000000009900000000009900000000009900000000009900000000009900000000009900000000009910000000009910000001998899100000000000000"),
    (3, "000000000000000999988000009999999900091000099920091000099920000000000910000000000900000000099000000000999000000999999900000999999900000000199990000000000990000000000490000000000290000000000910000000000910099000099800059999911000000000000000"),
    (2, "000000000000000099990000009900009900009900009900039900000080039900000080039900004400009990004400000099900000000099900000000069990000008800049900008800049900048800006690098800000090098800000090009900009900009900009900000099990000000000000000"),
    (4, "000000000000000000099000000000999000000005999000000005999000000007599000000070699000006600799000006600899000060000899000060000899000040000799000800000799000999999999990000000799000000000899000000000899000000000899000000000999000000000000000"),
    (7, "000000000000099999999990199999999980810000000900810000000900300000000900000000055900000000099200000000099000000000299000000000299000000000922000000000900000000001900000000009400000000009000000000009000000000019000000000193000000000000000000"),
    (1, "000000000000000092200000009990000000000090000000000090000000000090000000000090000000000090000000000090000000000090000000000090000000000090000000000090000000000090000000000090000000000090000000000090000000000094400000009999999000000000000000"),
    (4, "000000000000000000018900000000099900000000399900000004999900000005889900000003089900000030099900000030089900003300089900003300089900012200089900030000089900499999999994000000099900000000099900000000099900000000099900000000089900000000000000"),
    (5, "000000000000000009999990000119999990000990000000000990000000000990000000009999000000009999999000000002999900000000099990000000099990000000000890000000000090000000000090000000000190000000000910000000000910099000099300009999900000000000000000"),
    (0, "000000000000000099990000009900009900009900009900009900004480009900004480090000000080090000000080090000000080090000000080090000000080090000000080090000000080090000000080059900002280059900002280009900009900009900009900000099990000000000000000"),
    (2, "000000000000106688700000099999999000850000999800850000999800700000099900000000044900000000044900000000099200000000099000000000099000000000900000000009200000000077000000000090000000009900000090009900000090199999999940999999999900000000000000"),
    (6, "000000000000000000059990000009990000000009990000000090000000000090000000009900000000059909999900099900009990099900009990090000003390090000000090090000000090090000000090099900000090099900000090009900009900009900009900000099990000000000000000"),
    (1, "000000000000000720000000000081100000000018800000000019900000000009910000000009910000000009990000000009990000000009990000000009990000000009990000000009990000000109930000000019900000000039900000000038800000000182200000000810000000000000000000"),
    (9, "000000000000000099990000009900009900009900009900090000009990090000009990090000000090090000000090090000000090090000000090009900009990009999909990009999909990000000009900000000090000000000090000000009900000000009900000099990000000000000000000"),
)


def _digit_banks() -> tuple[tuple[int, np.ndarray], ...]:
    banks: list[tuple[int, np.ndarray]] = []
    for digit, packed in _DIGIT_TEMPLATES:
        values = np.array([int(char) for char in packed], np.float32).reshape(20, 12) / 9.0
        banks.append((int(digit), values))
    return tuple(banks)


_DIGIT_BANKS = _digit_banks()


def _drop_prefixed_numbers(
    batch: list[tuple[int | None, tuple[int, int, int], Image.Image]],
) -> list[tuple[int | None, tuple[int, int, int], Image.Image]]:
    """同一页上，6 紧挨着 61 或 64 时，6 是没读全的号，不能拿去覆盖真的第 6 小节。"""
    cleaned = []
    for index, (number, rank, piece) in enumerate(batch):
        if number is None:
            cleaned.append((number, rank, piece))
            continue
        text = str(number)
        truncated = False
        for other_index in range(max(0, index - 2), min(len(batch), index + 3)):
            if other_index == index:
                continue
            other = batch[other_index][0]
            if other is None:
                continue
            longer = str(other)
            # 1 和 18 会同时出现在有多小节休止的那一页，不能当成没读全。
            # 6 和 61、16 和 160 隔了几十小节，短的那个才是切剩的。
            if longer.startswith(text) and len(longer) > len(text) and abs(other - number) > 20:
                truncated = True
                break
        cleaned.append((None if truncated else number, rank, piece))
    return _drop_outlier_numbers(cleaned)


def _drop_outlier_numbers(
    batch: list[tuple[int | None, tuple[int, int, int], Image.Image]],
) -> list[tuple[int | None, tuple[int, int, int], Image.Image]]:
    """同一页上离群的号是没读全的。159 读成 59 时，留下旁边的 160。"""
    numbers = [number for number, _rank, _piece in batch if number is not None]
    if len(numbers) < 2:
        return batch
    ordered_values = sorted(numbers)
    median = ordered_values[len(ordered_values) // 2]
    if median >= 16:
        adjusted = []
        for number, item_rank, piece in batch:
            if number is None or number >= 10:
                adjusted.append((number, item_rank, piece))
                continue
            others = [other for other in numbers if other != number]
            nearest = min(abs(number - other) for other in others) if others else 0
            # 30 多小节那一页上的 6，是 36 被切掉了十位，不是真的第 6 小节。
            if nearest > 8:
                adjusted.append((None, item_rank, piece))
            else:
                adjusted.append((number, item_rank, piece))
        batch = adjusted
        numbers = [number for number, _rank, _piece in batch if number is not None]
        if len(numbers) < 2:
            return batch
    ordered = sorted(set(numbers))
    clusters: list[list[int]] = [[ordered[0]]]
    for number in ordered[1:]:
        if number - clusters[-1][-1] > 40:
            clusters.append([number])
        else:
            clusters[-1].append(number)
    if len(clusters) == 1:
        return batch

    def rank(cluster: list[int]) -> tuple[int, int]:
        return (max(len(str(item)) for item in cluster), len(cluster))

    kept = set(max(clusters, key=rank))
    return [
        (None, item_rank, piece) if number is not None and number not in kept else (number, item_rank, piece)
        for number, item_rank, piece in batch
    ]


def _measure_digit_ok(glyph: tuple) -> bool:
    """完整数字要够像。被谱线拆开再拼上的，相关低一截也认。"""
    digit, score, margin, merged = glyph[4], glyph[5], glyph[6], glyph[7]
    if digit is None:
        return False
    if score >= 0.74 and margin >= 0.045:
        return True
    if score >= 0.68 and margin >= 0.20:
        return True
    # 四线谱上的小节号更细，相关落在 0.74 下面，但和第二名仍差一截。
    if score >= 0.52 and margin >= 0.10 and 5 <= glyph[2] <= 8 and glyph[3] >= 11:
        return True
    return bool(merged and score >= 0.37 and margin >= 0.07)


def _vertical_gap(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if left[1] + left[3] < right[1]:
        return right[1] - (left[1] + left[3])
    if right[1] + right[3] < left[1]:
        return left[1] - (right[1] + right[3])
    return 0


def _glyph_from_pieces(
    labels: np.ndarray,
    members: list[tuple[int, int, int, int, int, int]],
) -> tuple[int, int, int, int, int, int | None, float, float]:
    x0 = min(piece[0] for piece in members)
    y0 = min(piece[1] for piece in members)
    x1 = max(piece[0] + piece[2] for piece in members)
    y1 = max(piece[1] + piece[3] for piece in members)
    width, height = x1 - x0, y1 - y0
    crop = np.zeros((height, width), bool)
    area = 0
    for x, y, piece_w, piece_h, _area, label in members:
        region = labels[y : y + piece_h, x : x + piece_w] == label
        crop[y - y0 : y - y0 + piece_h, x - x0 : x - x0 + piece_w] |= region
        area += int(region.sum())
    if height < 9 or height > 22 or width < 3 or width > 16 or area < 12:
        return (x0, y0, width, height, area, None, 0.0, 0.0)
    digit, score, margin = _classify_glyph(_fit_glyph(crop))
    if digit == 2 and _two_bowls(crop):
        # 这种 8 缩进模板后和 2 几乎重合，上下两个空腔才分得开。
        digit, score, margin = 8, max(float(score), 0.80), max(float(margin), 0.12)
    return (x0, y0, width, height, area, digit, float(score), float(margin))


def _stack_measure_digits(
    labels: np.ndarray,
    pieces: list[tuple[int, int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int, int | None, float, float, bool]]:
    """上下只隔几个像素、左右又叠在一起的，是同一位被谱线拆开了。

    两个都够高的数字不并，避免把 9 和 2 粘成一个字。
    """
    count = len(pieces)
    parent = list(range(count))
    box_x0 = [piece[0] for piece in pieces]
    box_y0 = [piece[1] for piece in pieces]
    box_x1 = [piece[0] + piece[2] for piece in pieces]
    box_y1 = [piece[1] + piece[3] for piece in pieces]
    full = [piece[3] >= 9 for piece in pieces]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            left, right = pieces[left_index], pieces[right_index]
            if left[3] >= 9 and right[3] >= 9:
                continue
            overlap = min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0])
            narrow = min(left[2], right[2])
            if narrow <= 0 or overlap * 2 < narrow:
                continue
            if _vertical_gap(left, right) > 4:
                continue
            left_root, right_root = find(left_index), find(right_index)
            if left_root == right_root or (full[left_root] and full[right_root]):
                continue
            x0 = min(box_x0[left_root], box_x0[right_root])
            y0 = min(box_y0[left_root], box_y0[right_root])
            x1 = max(box_x1[left_root], box_x1[right_root])
            y1 = max(box_y1[left_root], box_y1[right_root])
            if x1 - x0 > 16 or y1 - y0 > 22:
                continue
            parent[right_root] = left_root
            box_x0[left_root], box_y0[left_root] = x0, y0
            box_x1[left_root], box_y1[left_root] = x1, y1
            full[left_root] = full[left_root] or full[right_root]
    glyphs = []
    grouped: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(pieces[index])
    for members in grouped.values():
        union = _glyph_from_pieces(labels, members)
        tall = [piece for piece in members if piece[3] >= 9]
        merged = len(members) > 1
        if merged and tall:
            parent_glyph = _glyph_from_pieces(labels, [max(tall, key=lambda piece: piece[4])])
            if parent_glyph[5] is not None and parent_glyph[6] >= union[6]:
                glyphs.append((*parent_glyph, False))
                continue
        glyphs.append((*union, merged))
    return glyphs


def _read_measure_number(image: Image.Image, mask: np.ndarray) -> int | None:
    """先用谱面墨水读。只有一两位时，再用谱号上方更松的阈值补淡笔画。

    淡笔画读出更长的号才采用。三位已经读全就不再放宽，避免把 204 涂成 4。
    横杠并进第一位、把 211 读成 811 时，淡笔画分开了就用淡的那一版。
    """
    strict = _read_measure_number_on(image, mask)
    soft = _read_measure_number_on(image, _soft_number_mask(image, mask))
    if (
        strict is not None
        and soft is not None
        and strict >= 100
        and soft >= 100
        and str(strict)[0] == "8"
        and str(soft)[0] == "2"
        and str(strict)[1:] == str(soft)[1:]
    ):
        return soft
    if strict is not None and strict >= 100:
        return strict
    if soft is None:
        return strict
    # 墨水已经拒绝的号，不能用只多读出一位的淡笔画结果把它救回来。
    # 淡笔画要把 2 补成 25、把 6 补成 67 才算数。
    if strict is None:
        return soft if soft >= 10 else None
    if len(str(soft)) > len(str(strict)):
        return soft
    return strict


def _soft_number_mask(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    """灰纸上的数字比符头淡。只放宽左上角、谱线上方那一块。"""
    gray = np.array(ImageOps.grayscale(image.convert("RGB")))
    systems = legacy_systems(gray)
    if not systems:
        return mask
    staff = int(min(item[0] for item in systems))
    spacing = float(min(item[2] for item in systems))
    lighter = gray[gray >= np.percentile(gray, 60)]
    background = float(np.median(lighter)) if lighter.size else 255.0
    if background < 180.0:
        return mask
    level = min(background - 40.0, 175.0)
    faint = np.where(gray < level, np.uint8(255), np.uint8(0))
    y0 = max(0, int(staff - spacing * 2.6))
    y1 = max(y0, staff - 1)
    # 谱号和时间号会把小节号推过 96 像素。只放宽到能盖住三位数字。
    reach = max(96, int(round(spacing * 7.0)))
    softened = mask.copy()
    softened[y0:y1, :reach] = faint[y0:y1, :reach]
    return softened


def _read_measure_number_on(image: Image.Image, mask: np.ndarray) -> int | None:
    """只认谱线上方、靠左的一串数字。中间的高音和品位数字不算小节号。"""
    gray = np.array(ImageOps.grayscale(image.convert("RGB")))
    systems = legacy_systems(gray)
    if not systems:
        return None
    staff = float(min(item[0] for item in systems))
    spacing = float(min(item[2] for item in systems))
    ink = without_staff_lines(mask)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    pieces: list[tuple[int, int, int, int, int, int]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if y < staff - spacing * 2.6 or y + height / 2 > staff - 2:
            continue
        if height < 2 or height > 22 or width < 2 or width > 16 or area < 8:
            continue
        pieces.append((x, y, width, height, area, label))
    stacked = _stack_measure_digits(labels, pieces)
    glyphs: list[tuple[int, int, int, int, int | None, float, float, bool]] = []
    stubs: list[tuple[int, int, int, int]] = []
    sliver_right = -1
    for x, y, width, height, area, digit, score, margin, merged in stacked:
        # 拼进数字里的碎片不再当成被小节线切掉的半个字。
        if not merged and x <= 2 and width <= 6 and 8 <= height <= 22:
            sliver_right = max(sliver_right, x + width)
            if width <= 4:
                continue
        if height < 9 or height > 22 or width < 3 or width > 16 or area < 12:
            # 7 被切掉竖笔时，往往只剩左边或右边的一条横杠。
            if 2 <= height <= 6 and 6 <= width <= 16 and area >= 12:
                stubs.append((x, y, width, height))
            # 中间一位只剩半截时，不能把两边拼成一个更短的号。
            elif 7 <= height <= 8 and width >= 5 and area >= 16:
                glyphs.append((x, y, width, height, None, 0.40, 0.0, False))
            continue
        glyphs.append((x, y, width, height, digit, score, margin, merged))
    glyphs.sort(key=lambda item: item[0])
    if not glyphs:
        return None
    left_limit = max(80.0, min(image.width * 0.55, spacing * 12.0))
    start: int | None = None
    for index, glyph in enumerate(glyphs):
        if glyph[0] > left_limit:
            return None
        if not _measure_digit_ok(glyph):
            # 紧挨着的半个数字不能跳过，否则 35 会变成 5。离得远的污点才丢掉。
            nxt = glyphs[index + 1] if index + 1 < len(glyphs) else None
            if nxt is not None:
                gap = nxt[0] - (glyph[0] + glyph[2])
                if -2 <= gap <= max(12.0, spacing * 0.9):
                    return None
            if glyph[5] >= 0.50:
                return None
            continue
        start = index
        break
    if start is None:
        return None
    chosen = [glyphs[start]]
    # 左边还粘着被小节线切掉的半个数字时，不能把剩下的当成完整小节号。
    if sliver_right >= 0 and chosen[0][0] <= sliver_right + 6 and chosen[0][0] > sliver_right - chosen[0][2]:
        if not (chosen[0][5] >= 0.93 and chosen[0][0] <= 2):
            return None
    limit = max(12.0, spacing * 0.9)
    for glyph in glyphs[start + 1 :]:
        previous = chosen[-1]
        gap = glyph[0] - (previous[0] + previous[2])
        if gap > limit or gap < -2:
            break
        if abs((glyph[1] + glyph[3] / 2) - (previous[1] + previous[3] / 2)) > spacing * 0.75:
            break
        if not _measure_digit_ok(glyph):
            # 像一位数字但没读准，留下前面几位会把 35 读成 3。太淡的污点不算。
            if glyph[5] >= 0.32:
                return None
            break
        chosen.append(glyph)
        if len(chosen) == 3:
            break
    if len(chosen) == 3:
        last = chosen[-1]
        for glyph in glyphs:
            if glyph[0] <= last[0]:
                continue
            gap = glyph[0] - (last[0] + last[2])
            if gap <= limit and glyph[5] >= 0.62:
                return None
            break
    # 中间一位整个没了，剩下的两位隔开一截。不能把 153 读成 1。
    last = chosen[-1]
    peek = max(limit, spacing * 1.6)
    for glyph in glyphs:
        if glyph[0] <= last[0]:
            continue
        gap = glyph[0] - (last[0] + last[2])
        if gap > peek:
            break
        if glyph[3] >= 9 and glyph[5] >= 0.45:
            return None
        break
    value = 0
    for glyph in chosen:
        if glyph[4] is None:
            return None
        value = value * 10 + glyph[4]
    if _stub_beside(stubs, chosen[0], spacing) or _stub_beside(stubs, chosen[-1], spacing):
        return None
    return value


def _stub_beside(
    stubs: list[tuple[int, int, int, int]],
    glyph: tuple,
    spacing: float,
) -> bool:
    """数字紧左或紧右还有一条没拼上的横杠时，这一位是被切掉的。"""
    dx, dy, dw, dh = glyph[0], glyph[1], glyph[2], glyph[3]
    mid = dy + dh / 2.0
    limit = max(12.0, spacing * 0.9)
    for sx, sy, sw, sh in stubs:
        if abs((sy + sh / 2.0) - mid) > spacing * 0.75:
            continue
        if sx + sw <= dx + 2:
            gap = dx - (sx + sw)
        elif sx >= dx + dw - 2:
            gap = sx - (dx + dw)
        else:
            continue
        if -2 <= gap <= limit:
            return True
    return False


def _fit_glyph(crop: np.ndarray) -> np.ndarray:
    """把数字缩成和模板一样的 20×12，高度固定，左右居中。"""
    height, width = crop.shape
    scale = 18.0 / max(1, height)
    resized_w = max(1, int(round(width * scale)))
    patch = np.array(
        Image.fromarray(np.where(crop, np.uint8(255), np.uint8(0))).resize(
            (resized_w, 18), Image.Resampling.NEAREST
        )
    )
    canvas = np.zeros((20, 12), np.float32)
    left = max(0, (12 - patch.shape[1]) // 2)
    usable = min(12 - left, patch.shape[1])
    canvas[1:19, left : left + usable] = patch[:, :usable] > 0
    return canvas


def _two_bowls(crop: np.ndarray) -> bool:
    """上下两截左右都有墨、下半中间是空的。2 的底是实线，不是这种空腔。"""
    ink = crop.astype(bool)
    height, width = ink.shape
    if height < 12 or width < 4:
        return False
    def edges(part: np.ndarray) -> bool:
        if part.size == 0:
            return False
        side = max(1, width // 3)
        return bool(part[:, :side].any() and part[:, -side:].any())

    lower = ink[height // 2 :]
    middle = lower[:, width // 3 : max(width // 3 + 1, 2 * width // 3)]
    side = max(1, width // 4)
    hollow = float(middle.mean()) < 0.35 and bool(
        lower[:, :side].any() and lower[:, -side:].any()
    )
    # 2 的最后两行是通栏底线。8 的底部仍是空腔，不会铺满。
    tail = max(2, height // 5)
    bottom = ink[-tail:]
    if float(bottom.mean()) >= 0.55:
        return False
    return bool(edges(ink[: height // 3]) and edges(ink[2 * height // 3 :]) and hollow)


def _classify_glyph(canvas: np.ndarray) -> tuple[int | None, float, float]:
    scores: dict[int, float] = {}
    for digit, template in _DIGIT_BANKS:
        score = _glyph_corr(canvas, template)
        if score > scores.get(digit, -1.0):
            scores[digit] = score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, 0.0, 0.0
    best_digit, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    return best_digit, best_score, best_score - second


# 模板库摊成的矩阵（每行去均值、每行的模长），按库对象缓存；按曲子学的库每首新建一个，只留最近几个。
_MATRICES: dict[int, tuple[object, tuple[int, ...] | None, np.ndarray, np.ndarray]] = {}


def corr_all(canvas: np.ndarray, bank) -> list[float]:
    """canvas 和库里每个模板的 _glyph_corr（去均值后的归一化相关），按库的顺序。

    整个库摊成一个矩阵一次乘完，结果和逐个算一样；读谱时每个字要和几百个模板比，逐个算太慢。
    """
    entry = _MATRICES.get(id(bank))
    if entry is None or entry[0] is not bank:
        shapes = {template.shape for _label, template in bank}
        shape = next(iter(shapes)) if len(shapes) == 1 else None
        if shape is not None:
            flat = np.stack([template.astype(np.float64).ravel() for _label, template in bank])
            flat -= flat.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(flat, axis=1)
        else:
            flat, norms = np.zeros((0, 0)), np.zeros(0)
        if len(_MATRICES) >= 16:
            _MATRICES.clear()
        entry = (bank, shape, flat, norms)
        _MATRICES[id(bank)] = entry
    _bank, shape, flat, norms = entry
    if shape is None or canvas.shape != shape:
        return [_glyph_corr(canvas, template) for _label, template in bank]
    vector = canvas.astype(np.float64).ravel()
    vector = vector - vector.mean()
    denom = norms * float(np.linalg.norm(vector))
    safe = np.where(denom < 1e-6, 1.0, denom)
    return np.where(denom < 1e-6, 0.0, (flat @ vector) / safe).tolist()


def _glyph_corr(left: np.ndarray, right: np.ndarray) -> float:
    aa = left - float(left.mean())
    bb = right - float(right.mean())
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(aa.ravel(), bb.ravel()) / denom)
