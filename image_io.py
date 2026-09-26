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

Phones often store a photo in the sensor's native orientation and add an
EXIF Orientation tag saying how to rotate it for display. Pillow ignores
that tag, so open_image() applies it; otherwise such photos come out
sideways or upside down. pillow-heif and rawpy already apply it themselves
(pillow-heif resets the tag to 1 afterwards), so only JPEG and friends need
the transpose here.
"""

import os

import pillow_heif
import rawpy
from typing import Optional, Tuple

from PIL import ExifTags, Image, ImageOps

pillow_heif.register_heif_opener()

# Extensions decoded by rawpy rather than Pillow.
RAW_EXTENSIONS = ('.dng',)


def open_image(image_path: str,
               draft_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Open an image file as a PIL Image, upright per its EXIF orientation.

    Usable as a context manager, like Image.open().

    draft_size, if given, asks the decoder for a reduced-scale image no
    smaller than that size (see Image.draft). It has to be passed here
    rather than called on the result: draft() only works before the pixels
    are loaded, and rotating an image loads it. The size is treated as a
    bound on both sides, so pass a square to be independent of rotation.

    Raises:
        FileNotFoundError: If the file does not exist
        Exception: Whatever the underlying decoder raises for a bad file
    """
    if os.path.splitext(image_path)[1].lower() in RAW_EXTENSIONS:
        return _open_raw(image_path)
    img = Image.open(image_path)
    if draft_size is not None:
        img.draft('RGB', draft_size)
    return _apply_exif_orientation(img)


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Rotate/flip img upright according to its EXIF Orientation tag.

    An image that needs no change is returned as is, still lazily loaded
    and keeping its format. Otherwise the source is closed and a new,
    loaded image returned in its place.
    """
    try:
        orientation = img.getexif().get(ExifTags.Base.Orientation, 1)
    except Exception:
        # A malformed EXIF block should not make the photo unreadable.
        return img
    if orientation not in range(2, 9):
        return img
    with img:
        return ImageOps.exif_transpose(img)


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
