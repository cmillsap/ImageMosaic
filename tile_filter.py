"""
Tile Filter

Spots tiles that are not really photographs - screenshots, maps, documents,
receipts, memes - so they can be left out of a mosaic. A phone's camera
roll is full of them, and in a mosaic they show up as flat white rectangles
of text.

Two tests, either of which marks a tile as junk:

* A CLIP image model, run zero-shot: each tile's embedding is compared with
  text prompts describing junk ("a screenshot of a text message
  conversation") and ordinary photos ("a photo of a family"). The prompt
  embeddings are computed once by packaging/make_filter_prompts.py and
  committed; the image model is 176 MB, so it is downloaded the first time
  the filter is switched on rather than shipped in the installer.

* A file check for screen captures: a PNG at an exact phone, tablet or
  monitor resolution with no camera make or model in its metadata. CLIP
  misses screenshots that are mostly a photo - a lock screen showing its
  wallpaper, a shared Instagram story - and this catches those.

Tested on a 12,000-photo library, the model excluded about 4% of tiles;
in samples roughly four in five were plainly junk, the rest borderline
(museum placards, gravestones, retro computers with text on screen).

This module deliberately imports no PyQt: tile_loader's worker processes
import it too.
"""

import hashlib
import logging
import os
import sqlite3
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# The image half of OpenAI's CLIP ViT-B/32, as fp16 ONNX. fp16 was chosen
# over the smaller int8 and 4-bit exports because on a real library it gave
# exactly the same decisions as fp32, while int8 flipped about 8% of the
# borderline ones. Pinned to a commit so the file never changes underneath
# MODEL_SHA256.
MODEL_URL = ("https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
             "d15189d7028b43f1d3e65039190477f6af591c2a/onnx/vision_model_fp16.onnx")
MODEL_SHA256 = "35c4e0fb0aeee527dcde1693520b214a34424a786babd530f35366bad5844efd"
MODEL_SIZE = 176080659
MODEL_FILENAME = "clip-vit-b32-vision-fp16.onnx"

PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tile_filter_prompts.npz")

# A tile is junk when the junk prompts together take at least this share
# of the probability. 0.5 is the natural midpoint; on the test library
# everything below it was a real photo.
JUNK_THRESHOLD = 0.5

# CLIP's logit scale: cosine similarities are multiplied by this before
# the softmax.
LOGIT_SCALE = 100.0

# Part of every score-cache key. Bump it whenever the model, the prompts or
# the scoring changes, so scores from older code are not reused.
SCORER_VERSION = 1

CLIP_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

# Category reported for a tile caught by the file check.
SCREEN_CAPTURE = "Screenshot"

# Screen resolutions, portrait, as (width, height); landscape is matched
# too. Phones and tablets are Apple's native screenshot sizes plus common
# Android ones; monitors are the usual desktop resolutions.
SCREEN_SIZES = {
    # iPhone
    (640, 1136), (750, 1334), (828, 1792), (960, 2079), (1080, 1920),
    (1125, 2436), (1170, 2532), (1179, 2556), (1206, 2622), (1242, 2208),
    (1242, 2688), (1284, 2778), (1290, 2796), (1320, 2868),
    # iPad
    (1488, 2266), (1536, 2048), (1620, 2160), (1640, 2360), (1668, 2224),
    (1668, 2388), (2048, 2732),
    # Android
    (720, 1280), (720, 1600), (1080, 2160), (1080, 2340), (1080, 2400),
    (1440, 2560), (1440, 3040), (1440, 3120), (1440, 3200),
    # Monitors
    (768, 1366), (800, 1280), (900, 1440), (900, 1600), (1050, 1680),
    (1080, 2560), (1200, 1920), (1440, 2560), (1600, 2560), (1800, 2880),
    (1964, 3024), (2160, 3840),
}

# EXIF tags that only a camera writes.
_EXIF_MAKE = 271
_EXIF_MODEL = 272


@dataclass
class FilterConfig:
    """Where the filter's model and score cache live.

    Picklable, so it can travel with the tile loader's other settings.
    """
    model_path: str
    #: Directory for the score cache; None disables caching.
    cache_dir: Optional[str] = None
    threshold: float = JUNK_THRESHOLD


def model_path(model_dir: str) -> str:
    return os.path.join(model_dir, MODEL_FILENAME)


def is_model_installed(model_dir: str) -> bool:
    """True when the model has been downloaded to model_dir.

    Only checks the size: the hash was verified when it was downloaded,
    and rehashing 176 MB on every launch would be slow.
    """
    path = model_path(model_dir)
    return os.path.isfile(path) and os.path.getsize(path) == MODEL_SIZE


# ----------------------------------------------------------------------
# Tests that run on one image, in the worker processes
# ----------------------------------------------------------------------

def is_screen_capture(path: str) -> bool:
    """A PNG at an exact screen resolution, with no camera in its EXIF.

    Reads only the file's header, so it costs next to nothing.
    """
    if os.path.splitext(path)[1].lower() != ".png":
        return False
    try:
        with Image.open(path) as image:
            width, height = image.size
            if (min(width, height), max(width, height)) not in SCREEN_SIZES:
                return False
            exif = image.getexif()
    except Exception:                                        # noqa: BLE001
        return False
    return not (exif.get(_EXIF_MAKE) or exif.get(_EXIF_MODEL))


def is_junk(tile, threshold: float = JUNK_THRESHOLD) -> bool:
    """True when a tile's filter result marks it for exclusion."""
    result = getattr(tile, "filter_result", None)
    return result is not None and result[0] >= threshold


def clip_input(image: Image.Image) -> np.ndarray:
    """The image as CLIP sees it: 224x224 RGB, uint8.

    Resized so the short side is 224, then centre-cropped, as CLIP was
    trained. Kept as uint8 rather than normalised floats so it is a quarter
    the size to send back from a worker process.
    """
    image = image.convert("RGB")
    width, height = image.size
    scale = CLIP_SIZE / min(width, height)
    size = (max(CLIP_SIZE, round(width * scale)),
            max(CLIP_SIZE, round(height * scale)))
    image = image.resize(size, Image.Resampling.BICUBIC)
    left = (size[0] - CLIP_SIZE) // 2
    top = (size[1] - CLIP_SIZE) // 2
    image = image.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))
    return np.asarray(image, dtype=np.uint8)


# ----------------------------------------------------------------------
# The model, run in the main process
# ----------------------------------------------------------------------

class JunkScorer:
    """Scores batches of clip_input arrays with the CLIP image model.

    One instance, in one process: the model takes about 350 MB of memory
    once loaded, so it is not worth a copy per worker. Inference is about
    10 ms per tile, well under what decoding and analysing a tile costs.
    """

    def __init__(self, model_file: str, prompts_file: str = PROMPTS_PATH,
                 session=None):
        """
        Args:
            model_file: The downloaded ONNX model
            prompts_file: Prompt embeddings from make_filter_prompts.py
            session: Stands in for the ONNX session; for tests
        """
        prompts = np.load(prompts_file)
        self._prompt_embeddings = prompts["embeddings"]
        self._categories = prompts["categories"]
        self._junk = self._categories != ""

        if session is None:
            import onnxruntime
            session = onnxruntime.InferenceSession(
                model_file, providers=["CPUExecutionProvider"])
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def score(self, images: Sequence[np.ndarray]) -> List[Tuple[float, str]]:
        """(junk probability, likeliest junk category) for each image."""
        if not images:
            return []
        batch = np.stack(images).astype(np.float32) / 255.0
        batch = ((batch - CLIP_MEAN) / CLIP_STD).transpose(0, 3, 1, 2)
        embeddings = self._session.run(None, {self._input_name: batch})[0]
        embeddings = embeddings.astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        logits = LOGIT_SCALE * embeddings @ self._prompt_embeddings.T
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        junk = probabilities[:, self._junk]
        junk_categories = self._categories[self._junk]
        return [(float(row.sum()), str(junk_categories[row.argmax()]))
                for row in junk]


class ScoreCache:
    """Remembers each tile's score between runs.

    Keyed on the file's path and modification time, like the tile cache, so
    an edited photo is scored again. Only ever opened by one process: the
    workers hand their results back rather than writing here themselves.
    """

    def __init__(self, cache_dir: Optional[str]):
        self._db = None
        if not cache_dir:
            return
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self._db = sqlite3.connect(
                os.path.join(cache_dir, "tile_filter_scores.sqlite"))
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, "
                "score REAL NOT NULL, category TEXT NOT NULL)")
        except (OSError, sqlite3.Error) as e:
            logger.warning("Tile filter cache unavailable: %s", e)
            self._db = None

    @staticmethod
    def _key(path: str) -> Optional[str]:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        return f"{SCORER_VERSION}:{os.path.abspath(path)}:{mtime}"

    def get_many(self, paths: Iterable[str]) -> Dict[str, Tuple[float, str]]:
        """Cached (score, category) for whichever of paths have one."""
        if self._db is None:
            return {}
        keys = {key: path for path in paths
                if (key := self._key(path)) is not None}
        found = {}
        try:
            for key, score, category in self._db.execute(
                    "SELECT key, score, category FROM scores"):
                if key in keys:
                    found[keys[key]] = (score, category)
        except sqlite3.Error as e:
            logger.warning("Tile filter cache read failed: %s", e)
        return found

    def put_many(self, scores: Dict[str, Tuple[float, str]]) -> None:
        if self._db is None or not scores:
            return
        rows = [(key, score, category)
                for path, (score, category) in scores.items()
                if (key := self._key(path)) is not None]
        try:
            with self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO scores VALUES (?, ?, ?)", rows)
        except sqlite3.Error as e:
            logger.warning("Tile filter cache write failed: %s", e)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------

class DownloadCancelled(Exception):
    """Raised by download_model when should_cancel returns True."""


def download_model(model_dir: str,
                   progress_callback: Optional[Callable[[int, int], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None,
                   url: str = MODEL_URL,
                   sha256: str = MODEL_SHA256,
                   chunk_size: int = 1 << 20) -> str:
    """Download the model into model_dir and return its path.

    Writes to a .part file and renames it only once the hash matches, so
    an interrupted or corrupted download never looks installed.

    Args:
        model_dir: Directory to install into; created if missing
        progress_callback: Called as (bytes so far, total bytes)
        should_cancel: Polled between chunks
        url, sha256: Overridable for tests

    Raises:
        DownloadCancelled: If should_cancel returned True
        ValueError: If the downloaded file's hash does not match
        OSError: For network and disk failures
    """
    os.makedirs(model_dir, exist_ok=True)
    final_path = model_path(model_dir)
    part_path = final_path + ".part"
    digest = hashlib.sha256()

    try:
        with urllib.request.urlopen(url, timeout=60) as response, \
                open(part_path, "wb") as out:
            total = int(response.headers.get("Content-Length") or MODEL_SIZE)
            done = 0
            while True:
                if should_cancel is not None and should_cancel():
                    raise DownloadCancelled()
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress_callback is not None:
                    progress_callback(done, total)

        if digest.hexdigest() != sha256:
            raise ValueError("The downloaded model is corrupt (checksum "
                             "mismatch). Please try again.")
        os.replace(part_path, final_path)
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)

    return final_path
