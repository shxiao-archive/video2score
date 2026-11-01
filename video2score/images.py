"""读写图片。cv2.imread/imwrite 碰到中文路径会失败（返回 None / False），这里绕开文件名这一层。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path | str) -> np.ndarray | None:
    """读成灰度图。文件不存在或损坏时返回 None，和 cv2.imread 一样。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def write_image(path: Path | str, image: np.ndarray) -> None:
    ok, data = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise ValueError(f"编码图片失败：{path}")
    data.tofile(str(path))
