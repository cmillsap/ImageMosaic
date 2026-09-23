"""
Image Loading

One entry point, open_image(), for every place that decodes a photo, so a
format needs supporting only once.

Most formats go straight to Pillow. HEIC/HEIF (the iPhone default) is
handled by pillow-heif, which registers itself as a Pillow plugin when this
module is imported - and because tile_loader's workers are fresh
interpreters on Windows, it matters that the registration happens here, in
a module they import, rather than once in the GUI process.

DNG cannot go through Pillow: Pillow opens it but returns only the ~160px
embedded thumbnail. rawpy (LibRaw) develops the raw sensor data instead.
"""

import os

import pillow_heif
import rawpy
from PIL import Image

pillow_heif.register_heif_opener()

# Extensions decoded by rawpy rather than Pillow.
RAW_EXTENSIONS = ('.dng',)


def open_image(image_path: str) -> Image.Image:
    """Open an image file as a PIL Image.

    Usable as a context manager, like Image.open(). For DNG the result is
    already decoded RGB, so a following draft() call is a harmless no-op.

    Raises:
        FileNotFoundError: If the file does not exist
        Exception: Whatever the underlying decoder raises for a bad file
    """
    if os.path.splitext(image_path)[1].lower() in RAW_EXTENSIONS:
        return _open_raw(image_path)
    return Image.open(image_path)


def _open_raw(image_path: str) -> Image.Image:
    """Develop a camera RAW file to 8-bit RGB.

    half_size skips demosaicing by treating each 2x2 Bayer block as one
    pixel, giving a quarter-resolution image in a fraction of the time. A
    12 MP file still yields 3 MP, far more than any tile or guide cell
    needs, and every caller downscales well below that anyway.
    use_camera_wb matches the white balance the photographer saw; LibRaw's
    default (a fixed daylight balance) visibly shifts indoor shots.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=True,
                              output_bps=8)
    return Image.fromarray(rgb)
