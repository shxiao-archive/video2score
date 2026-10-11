"""生成 video2score 的图标（video2score.ico、video2score.icns、video2score.png）：蓝色圆角方块上一个白色八分音符。

python packaging/icons/make_icons.py。只用 Pillow 画图形，不用字体。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
SIZE = 1024


def draw() -> Image.Image:
    scale = 4  # 先画大再缩小，边缘平滑
    side = SIZE * scale
    image = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    pen = ImageDraw.Draw(image)
    margin = int(side * 0.06)
    pen.rounded_rectangle((margin, margin, side - margin, side - margin), radius=int(side * 0.2), fill=(47, 111, 222, 255))
    white = (255, 255, 255, 255)
    # 五线谱的三条线（淡一点：和底色混出来的浅蓝，不用透明色，免得图标上透出洞来）。
    for fraction in (0.36, 0.5, 0.64):
        y = int(side * fraction)
        pen.rectangle((int(side * 0.18), y - side // 160, int(side * 0.82), y + side // 160), fill=(126, 163, 234, 255))
    # 符头：斜一点的椭圆。
    head = Image.new("RGBA", (int(side * 0.26), int(side * 0.18)), (0, 0, 0, 0))
    ImageDraw.Draw(head).ellipse((0, 0, head.width - 1, head.height - 1), fill=white)
    head = head.rotate(22, expand=True, resample=Image.BICUBIC)
    head_x, head_y = int(side * 0.28), int(side * 0.60)
    image.alpha_composite(head, (head_x, head_y))
    # 符干。
    stem_x = head_x + head.width - int(side * 0.03)
    top = int(side * 0.20)
    stem = int(side * 0.035)
    pen.rectangle((stem_x - stem, top, stem_x, head_y + head.height * 2 // 5), fill=white)
    # 符尾：从符干顶上往右下弯的一道粗弧。
    reach = int(side * 0.20)
    pen.arc((stem_x - reach - stem // 2, top, stem_x + reach, top + 2 * reach), start=-90, end=20, fill=white, width=int(side * 0.05))
    return image.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    image = draw()
    image.save(HERE / "video2score.png")
    image.save(HERE / "video2score.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    image.save(HERE / "video2score.icns")


if __name__ == "__main__":
    main()
