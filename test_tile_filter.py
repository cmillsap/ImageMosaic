"""
Tests for the Tile Filter and its wiring into tile loading.

The CLIP model is 176 MB and downloaded on demand, so almost everything
here runs against a stand-in scorer. Set TILE_FILTER_MODEL to the path of a
downloaded model to also run the tests that use the real one.
"""

import hashlib
import os
import shutil
import tempfile
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

import tile_filter
import tile_loader
from tile_database import TileDatabase
from tile_filter import (
    CLIP_SIZE,
    SCREEN_CAPTURE,
    DownloadCancelled,
    FilterConfig,
    JunkScorer,
    ScoreCache,
    clip_input,
    download_model,
    is_junk,
    is_screen_capture,
)
from tile_loader import PARALLEL_THRESHOLD, FilterUnavailable, analyse_paths
from tile_preprocessor import PreprocessorConfig

TILE = 10
REAL_MODEL = os.environ.get("TILE_FILTER_MODEL")

RED = (230, 20, 20)
BLUE = (20, 20, 230)


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def config():
    return PreprocessorConfig(target_width=TILE, target_height=TILE,
                              enable_face_detection=False,
                              enable_saliency=False)


class FakeScorer:
    """Stands in for JunkScorer: red tiles are documents, others photos."""

    calls = []

    def __init__(self, model_file):
        self.model_file = model_file

    def score(self, images):
        FakeScorer.calls.append(len(images))
        results = []
        for image in images:
            r, g, b = image.reshape(-1, 3).mean(axis=0)
            results.append((0.9, "Document") if r > 200 and b < 50
                           else (0.1, "Screenshot"))
        return results


@pytest.fixture
def fake_scorer(monkeypatch):
    FakeScorer.calls = []
    monkeypatch.setattr(tile_loader, "JunkScorer", FakeScorer)
    return FakeScorer


def make_tiles(folder, colors, ext=".jpg", size=(30, 30)):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, color in enumerate(colors):
        path = os.path.join(folder, f"t{i:04d}{ext}")
        Image.new("RGB", size, color).save(path)
        paths.append(path)
    return paths


def filter_config(temp_dir, **kwargs):
    return FilterConfig(model_path=os.path.join(temp_dir, "model.onnx"),
                        cache_dir=os.path.join(temp_dir, "cache"), **kwargs)


# ============================================================================
# Screen captures
# ============================================================================

class TestScreenCapture:

    def test_phone_screenshot_png(self, temp_dir):
        path = os.path.join(temp_dir, "shot.png")
        Image.new("RGB", (828, 1792)).save(path)
        assert is_screen_capture(path)

    def test_landscape_matches_too(self, temp_dir):
        path = os.path.join(temp_dir, "shot.png")
        Image.new("RGB", (1920, 1080)).save(path)
        assert is_screen_capture(path)

    def test_jpeg_at_screen_size_is_not(self, temp_dir):
        """Camera photos are JPEG or HEIC; screenshots are PNG."""
        path = os.path.join(temp_dir, "photo.jpg")
        Image.new("RGB", (1080, 1920)).save(path)
        assert not is_screen_capture(path)

    def test_png_at_other_size_is_not(self, temp_dir):
        """Astrophotography and edited exports are PNG at odd sizes."""
        path = os.path.join(temp_dir, "nebula.png")
        Image.new("RGB", (3708, 2088)).save(path)
        assert not is_screen_capture(path)

    def test_png_from_a_camera_is_not(self, temp_dir):
        path = os.path.join(temp_dir, "camera.png")
        exif = Image.Exif()
        exif[271] = "Canon"
        Image.new("RGB", (1080, 1920)).save(path, exif=exif)
        assert not is_screen_capture(path)

    def test_unreadable_file_is_not(self, temp_dir):
        path = os.path.join(temp_dir, "broken.png")
        with open(path, "wb") as f:
            f.write(b"not a png")
        assert not is_screen_capture(path)


# ============================================================================
# Model input
# ============================================================================

class TestClipInput:

    @pytest.mark.parametrize("size", [(640, 480), (480, 640), (50, 30),
                                      (224, 224)])
    def test_always_224_square_uint8(self, size):
        array = clip_input(Image.new("RGB", size, (1, 2, 3)))
        assert array.shape == (CLIP_SIZE, CLIP_SIZE, 3)
        assert array.dtype == np.uint8

    def test_centre_crops_the_long_side(self):
        img = Image.new("RGB", (900, 300), (255, 0, 0))
        img.paste((0, 0, 255), (300, 0, 600, 300))
        array = clip_input(img)
        assert tuple(array[112, 112]) == (0, 0, 255)

    def test_converts_to_rgb(self):
        array = clip_input(Image.new("L", (300, 300), 128))
        assert array.shape[2] == 3


# ============================================================================
# Scorer
# ============================================================================

class FakeSession:
    """Returns a fixed embedding for every image in the batch."""

    def __init__(self, embedding):
        self.embedding = embedding
        self.batches = []

    def get_inputs(self):
        class Input:
            name = "pixel_values"
        return [Input()]

    def run(self, _outputs, feed):
        batch = feed["pixel_values"]
        self.batches.append(batch.shape)
        return [np.tile(self.embedding, (batch.shape[0], 1))]


class TestJunkScorer:

    @pytest.fixture
    def prompts(self):
        data = np.load(tile_filter.PROMPTS_PATH)
        return data["embeddings"], data["categories"], data["prompts"]

    def test_prompt_file_is_well_formed(self, prompts):
        embeddings, categories, texts = prompts
        assert embeddings.shape == (len(categories), 512)
        assert len(texts) == len(categories)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4)
        assert {"Screenshot", "Map", "Document", "Graphic", ""} == set(categories)

    def test_image_like_a_junk_prompt_scores_high(self, prompts):
        embeddings, categories, texts = prompts
        i = list(texts).index("a road map")
        scorer = JunkScorer("unused", session=FakeSession(embeddings[i]))
        [(score, category)] = scorer.score([np.zeros((224, 224, 3), np.uint8)])
        assert score > 0.9
        assert category == "Map"

    def test_image_like_a_photo_prompt_scores_low(self, prompts):
        embeddings, _, texts = prompts
        i = list(texts).index("a photo of a dog")
        scorer = JunkScorer("unused", session=FakeSession(embeddings[i]))
        [(score, _)] = scorer.score([np.zeros((224, 224, 3), np.uint8)])
        assert score < 0.1

    def test_batches_are_normalised_nchw_floats(self, prompts):
        session = FakeSession(prompts[0][0])
        JunkScorer("unused", session=session).score(
            [np.zeros((224, 224, 3), np.uint8)] * 3)
        assert session.batches == [(3, 3, 224, 224)]

    def test_empty_batch(self, prompts):
        scorer = JunkScorer("unused", session=FakeSession(prompts[0][0]))
        assert scorer.score([]) == []

    def test_is_junk_uses_the_threshold(self):
        class Tile:
            filter_result = (0.6, "Map")
        assert is_junk(Tile(), 0.5)
        assert not is_junk(Tile(), 0.7)
        Tile.filter_result = None
        assert not is_junk(Tile(), 0.5)


@pytest.mark.skipif(not REAL_MODEL, reason="set TILE_FILTER_MODEL to run")
class TestRealModel:

    def test_a_page_of_text_is_a_document(self):
        page = Image.new("RGB", (600, 800), "white")
        draw = ImageDraw.Draw(page)
        for y in range(40, 760, 24):
            draw.text((40, y), "Invoice total due on receipt  $1,234.00 "
                      "account number 000123", fill="black")
        [(score, category)] = JunkScorer(REAL_MODEL).score([clip_input(page)])
        assert score >= tile_filter.JUNK_THRESHOLD
        assert category in ("Document", "Graphic", "Screenshot")


# ============================================================================
# Score cache
# ============================================================================

class TestScoreCache:

    def test_round_trip(self, temp_dir):
        path = make_tiles(temp_dir, [RED])[0]
        cache = ScoreCache(os.path.join(temp_dir, "cache"))
        cache.put_many({path: (0.7, "Map")})
        cache.close()

        cache = ScoreCache(os.path.join(temp_dir, "cache"))
        assert cache.get_many([path]) == {path: (0.7, "Map")}

    def test_editing_a_file_invalidates_it(self, temp_dir):
        path = make_tiles(temp_dir, [RED])[0]
        cache = ScoreCache(os.path.join(temp_dir, "cache"))
        cache.put_many({path: (0.7, "Map")})
        later = time.time() + 10
        os.utime(path, (later, later))
        assert cache.get_many([path]) == {}

    def test_disabled_without_a_directory(self, temp_dir):
        path = make_tiles(temp_dir, [RED])[0]
        cache = ScoreCache(None)
        cache.put_many({path: (0.7, "Map")})
        assert cache.get_many([path]) == {}


# ============================================================================
# Download
# ============================================================================

class TestDownload:

    @pytest.fixture
    def source(self, temp_dir):
        path = os.path.join(temp_dir, "source.onnx")
        data = os.urandom(300_000)
        with open(path, "wb") as f:
            f.write(data)
        url = "file:///" + path.replace("\\", "/")
        return url, hashlib.sha256(data).hexdigest(), data

    def test_installs_and_reports_progress(self, temp_dir, source):
        url, digest, data = source
        progress = []
        target = os.path.join(temp_dir, "models")
        path = download_model(target, lambda d, t: progress.append((d, t)),
                              url=url, sha256=digest, chunk_size=65536)

        assert path == tile_filter.model_path(target)
        with open(path, "rb") as f:
            assert f.read() == data
        assert progress[-1] == (len(data), len(data))
        assert os.listdir(target) == [tile_filter.MODEL_FILENAME]

    def test_bad_checksum_leaves_nothing_behind(self, temp_dir, source):
        url, _, _ = source
        target = os.path.join(temp_dir, "models")
        with pytest.raises(ValueError):
            download_model(target, url=url, sha256="0" * 64)
        assert os.listdir(target) == []

    def test_cancel_leaves_nothing_behind(self, temp_dir, source):
        url, digest, _ = source
        target = os.path.join(temp_dir, "models")
        with pytest.raises(DownloadCancelled):
            download_model(target, should_cancel=lambda: True,
                           url=url, sha256=digest)
        assert os.listdir(target) == []

    def test_partial_file_does_not_count_as_installed(self, temp_dir):
        with open(tile_filter.model_path(temp_dir), "wb") as f:
            f.write(b"truncated")
        assert not tile_filter.is_model_installed(temp_dir)


# ============================================================================
# Tile loading with the filter on
# ============================================================================

class TestLoadingWithFilter:

    def test_off_by_default(self, temp_dir, config, fake_scorer):
        paths = make_tiles(temp_dir, [RED, BLUE])
        tiles = analyse_paths(paths, config, max_workers=1)
        assert all(t.filter_result is None for t in tiles)
        assert fake_scorer.calls == []

    def test_scores_every_tile(self, temp_dir, config, fake_scorer):
        paths = make_tiles(os.path.join(temp_dir, "t"), [RED, BLUE, RED])
        tiles = analyse_paths(paths, config, max_workers=1,
                              filter_config=filter_config(temp_dir))
        results = {os.path.basename(t.image_path): t.filter_result
                   for t in tiles}
        assert results == {"t0000.jpg": (0.9, "Document"),
                           "t0001.jpg": (0.1, "Screenshot"),
                           "t0002.jpg": (0.9, "Document")}
        assert all(t.clip_input is None for t in tiles)

    def test_second_run_uses_the_cache(self, temp_dir, config, fake_scorer):
        paths = make_tiles(os.path.join(temp_dir, "t"), [RED, BLUE])
        cfg = filter_config(temp_dir)
        analyse_paths(paths, config, max_workers=1, filter_config=cfg)
        fake_scorer.calls = []

        tiles = analyse_paths(paths, config, max_workers=1, filter_config=cfg)
        assert fake_scorer.calls == []
        assert sorted(t.filter_result for t in tiles) == [
            (0.1, "Screenshot"), (0.9, "Document")]

    def test_screen_captures_skip_the_model(self, temp_dir, config, fake_scorer):
        paths = make_tiles(os.path.join(temp_dir, "t"), [BLUE], ext=".png",
                           size=(1170, 2532))
        [tile] = analyse_paths(paths, config, max_workers=1,
                               filter_config=filter_config(temp_dir))
        assert tile.filter_result == (1.0, SCREEN_CAPTURE)
        assert fake_scorer.calls == []

    def test_scores_in_batches(self, temp_dir, config, fake_scorer):
        count = tile_loader.SCORE_BATCH_SIZE + 5
        paths = make_tiles(os.path.join(temp_dir, "t"), [BLUE] * count)
        analyse_paths(paths, config, max_workers=1,
                      filter_config=filter_config(temp_dir))
        assert fake_scorer.calls == [tile_loader.SCORE_BATCH_SIZE, 5]

    def test_parallel_path(self, temp_dir, config, fake_scorer):
        colors = [RED if i % 2 else BLUE for i in range(PARALLEL_THRESHOLD + 4)]
        paths = make_tiles(os.path.join(temp_dir, "t"), colors)
        tiles = analyse_paths(paths, config, max_workers=2,
                              filter_config=filter_config(temp_dir))
        assert len(tiles) == len(paths)
        junk = sum(1 for t in tiles if is_junk(t))
        assert junk == colors.count(RED)

    def test_missing_model_fails_loudly(self, temp_dir, config):
        """A filter that silently does nothing would look like it works."""
        paths = make_tiles(os.path.join(temp_dir, "t"), [RED])
        with pytest.raises(FilterUnavailable):
            analyse_paths(paths, config, max_workers=1,
                          filter_config=filter_config(temp_dir))

    def test_missing_model_fails_before_any_tile_is_analysed(
            self, temp_dir, config, monkeypatch):
        """Otherwise the pool would grind through the whole library first."""
        analysed = []
        monkeypatch.setattr(tile_loader, "_analyse_one",
                            lambda *a: analysed.append(a))
        paths = make_tiles(os.path.join(temp_dir, "t"), [RED, BLUE])
        with pytest.raises(FilterUnavailable):
            analyse_paths(paths, config, max_workers=1,
                          filter_config=filter_config(temp_dir))
        assert analysed == []

    def test_missing_model_is_not_retried_serially(self, temp_dir, config,
                                                   monkeypatch):
        serial = []
        real_serial = tile_loader._analyse_serial
        monkeypatch.setattr(tile_loader, "_analyse_serial",
                            lambda *a, **k: serial.append(1) or real_serial(*a, **k))
        paths = make_tiles(os.path.join(temp_dir, "t"),
                           [RED] * (PARALLEL_THRESHOLD + 1))
        with pytest.raises(FilterUnavailable):
            analyse_paths(paths, config, max_workers=2,
                          filter_config=filter_config(temp_dir))
        assert serial == []


class TestDatabaseExclusion:

    def test_junk_is_excluded_and_listed(self, temp_dir, config, fake_scorer):
        folder = os.path.join(temp_dir, "t")
        paths = make_tiles(folder, [RED, BLUE, BLUE])
        database = TileDatabase()
        loaded = database.load_tiles_parallel(
            folder, config, max_workers=1,
            filter_config=filter_config(temp_dir))

        assert loaded == 2
        assert len(database) == 2
        assert database.excluded_tiles == [(paths[0], "Document")]

    def test_threshold_is_respected(self, temp_dir, config, fake_scorer):
        folder = os.path.join(temp_dir, "t")
        make_tiles(folder, [RED, BLUE])
        database = TileDatabase()
        loaded = database.load_tiles_parallel(
            folder, config, max_workers=1,
            filter_config=filter_config(temp_dir, threshold=0.95))
        assert loaded == 2
        assert database.excluded_tiles == []

    def test_clear_forgets_exclusions(self, temp_dir, config, fake_scorer):
        folder = os.path.join(temp_dir, "t")
        make_tiles(folder, [RED, BLUE])
        database = TileDatabase()
        database.load_tiles_parallel(folder, config, max_workers=1,
                                     filter_config=filter_config(temp_dir))
        database.clear()
        assert database.excluded_tiles == []
