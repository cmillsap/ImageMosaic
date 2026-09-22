"""Generates packaging/icon.ico: a photo-mosaic grid rendered as an app icon.

Re-run after editing PALETTE; the .ico is committed so a normal build does
not need this script.
"""
import os

from PIL import Image, ImageDraw

# Tiles of one image sampled from a warm-to-cool sweep, so the icon reads as
# "many photos making one picture" even at 16px.
PALETTE = [
    (232, 122, 66), (240, 168, 78), (246, 205, 108), (214, 96, 84),
    (176, 88, 122), (118, 92, 156), (74, 110, 168), (62, 148, 172),
    (86, 172, 140), (140, 190, 108), (206, 214, 118), (238, 180, 96),
]


def build(size=512, grid=6, gap_ratio=0.055):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = round(size * 0.055)
    radius = round(size * 0.16)
    draw.rounded_rectangle([margin, margin, size - margin - 1, size - margin - 1],
                           radius=radius, fill=(30, 34, 44, 255))

    inner = margin + round(size * 0.055)
    span = size - inner * 2
    gap = max(1, round(span * gap_ratio))
    cell = (span - gap * (grid - 1)) / grid
    tile_radius = max(1, round(cell * 0.18))

    for row in range(grid):
        for col in range(grid):
            x0 = inner + col * (cell + gap)
            y0 = inner + row * (cell + gap)
            # Diagonal walk through the palette keeps neighbouring tiles
            # distinct while the overall image stays a smooth gradient.
            colour = PALETTE[(row * 2 + col * 3) % len(PALETTE)]
            shade = 1.0 - 0.22 * (row / max(1, grid - 1))
            colour = tuple(round(c * shade) for c in colour)
            draw.rounded_rectangle([round(x0), round(y0),
                                    round(x0 + cell) - 1, round(y0 + cell) - 1],
                                   radius=tile_radius, fill=colour + (255,))
    return img


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    icon = build()
    out = os.path.join(here, "icon.ico")
    icon.save(out, sizes=[(256, 256), (128, 128), (64, 64),
                          (48, 48), (32, 32), (16, 16)])
    print("wrote", out)


if __name__ == "__main__":
    main()
