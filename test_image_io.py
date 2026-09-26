"""
Tests for image_io.

The real-photo cases need test.HEIC and test.DNG in the project root (they
are gitignored, like face.jpg) and skip without them. They check full
resolution rather than just "it opened", because a DNG that silently falls
back to its embedded thumbnail still opens fine.
"""

import os
import shutil
import tempfile

import pytest
from PIL import ExifTags, Image

from guide_image import GuideImage
from image_io import open_image
from tile_analyzer import ImageTileAnalyzer
from tile_database import TileDatabase
import tile_loader
import tile_preprocessor
from tile_preprocessor import (PreprocessorConfig, TilePreprocessor,
                               TilePreprocessorCache)

HERE = os.path.dirname(__file__)
HEIC_PHOTO = os.path.join(HERE, 'test.HEIC')
DNG_PHOTO = os.path.join(HERE, 'test.DNG')

requires_heic = pytest.mark.skipif(
    not os.path.exists(HEIC_PHOTO),
    reason="test.HEIC fixture not present in project root"
)
requires_dng = pytest.mark.skipif(
    not os.path.exists(DNG_PHOTO),
    reason="test.DNG fixture not present in project root"
)

both_photos = pytest.mark.parametrize("path", [
    pytest.param(HEIC_PHOTO, marks=requires_heic, id="heic"),
    pytest.param(DNG_PHOTO, marks=requires_dng, id="dng"),
])


RED = (255, 0, 0)
BLUE = (0, 0, 255)


def _save_oriented_jpeg(path, orientation, size=(40, 20)):
    """A JPEG stored wide, left half red and right half blue, tagged with
    the given EXIF Orientation, the way a phone saves a sideways shot."""
    w, h = size
    img = Image.new('RGB', size, BLUE)
    img.paste(RED, (0, 0, w // 2, h))
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    img.save(path, 'JPEG', quality=95, exif=exif)


def _is_close(pixel, colour, tolerance=40):
    return all(abs(a - b) <= tolerance for a, b in zip(pixel, colour))


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


class TestOpenImage:

    def test_png_still_goes_through_pillow(self, temp_dir):
        path = os.path.join(temp_dir, 'a.png')
        Image.new('RGB', (30, 20), (10, 20, 30)).save(path)
        with open_image(path) as img:
            assert img.format == 'PNG'
            assert img.size == (30, 20)

    def test_missing_dng_raises_file_not_found(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            open_image(os.path.join(temp_dir, 'missing.dng'))

    def test_corrupt_dng_raises(self, temp_dir):
        path = os.path.join(temp_dir, 'bad.dng')
        with open(path, 'wb') as f:
            f.write(b'not a raw file')
        with pytest.raises(Exception):
            open_image(path)

    @requires_heic
    def test_heic_opens_at_full_resolution(self):
        with open_image(HEIC_PHOTO) as img:
            assert img.mode == 'RGB'
            assert min(img.size) >= 2000

    @requires_dng
    def test_dng_opens_well_above_thumbnail_size(self):
        """Pillow alone yields a ~160px thumbnail; half-size develop is
        a quarter of the sensor, still thousands of pixels."""
        with open_image(DNG_PHOTO) as img:
            assert img.mode == 'RGB'
            assert min(img.size) >= 1000

    @requires_dng
    def test_dng_accepts_draft(self):
        """preprocess_image() calls draft() on whatever open_image returns."""
        with open_image(DNG_PHOTO) as img:
            img.draft('RGB', (500, 500))
            assert min(img.size) >= 1000

    @requires_dng
    def test_dng_colour_is_plausible(self):
        """Guards against a white-balance or bit-depth mistake, which shows
        up as a near-black, near-white or strongly tinted image."""
        with open_image(DNG_PHOTO) as img:
            small = img.resize((32, 32))
        r, g, b = [sum(c) / len(c) for c in zip(*small.getdata())]
        assert 30 < (r + g + b) / 3 < 225
        assert max(r, g, b) - min(r, g, b) < 60


class TestExifOrientation:

    def test_rotate_90_tag_is_applied(self, temp_dir):
        """Orientation 6 means "rotate 90 degrees clockwise to display", so
        the stored left (red) half ends up on top."""
        path = os.path.join(temp_dir, 'phone.jpg')
        _save_oriented_jpeg(path, 6)
        with open_image(path) as img:
            assert img.size == (20, 40)
            img = img.convert('RGB')
            assert _is_close(img.getpixel((10, 5)), RED)
            assert _is_close(img.getpixel((10, 35)), BLUE)

    def test_rotate_180_tag_is_applied(self, temp_dir):
        path = os.path.join(temp_dir, 'phone.jpg')
        _save_oriented_jpeg(path, 3)
        with open_image(path) as img:
            assert img.size == (40, 20)
            assert _is_close(img.convert('RGB').getpixel((35, 10)), RED)

    def test_rotated_result_is_not_rotated_again(self, temp_dir):
        path = os.path.join(temp_dir, 'phone.jpg')
        _save_oriented_jpeg(path, 6)
        with open_image(path) as img:
            assert img.getexif().get(ExifTags.Base.Orientation, 1) == 1

    def test_upright_image_is_left_lazy(self, temp_dir):
        """No transpose needed means no copy: the image keeps its format and
        remains undecoded, so draft() still works for other callers."""
        path = os.path.join(temp_dir, 'upright.jpg')
        _save_oriented_jpeg(path, 1)
        with open_image(path) as img:
            assert img.format == 'JPEG'
            assert img.size == (40, 20)

    def test_draft_size_applies_before_rotation(self, temp_dir):
        """A large JPEG decodes at reduced scale and still comes out
        rotated; drafting after the transpose would be a no-op."""
        path = os.path.join(temp_dir, 'big.jpg')
        _save_oriented_jpeg(path, 6, size=(1600, 800))
        with open_image(path, draft_size=(200, 200)) as img:
            assert img.size == (200, 400)

    def test_draft_size_is_ignored_for_png(self, temp_dir):
        path = os.path.join(temp_dir, 'a.png')
        Image.new('RGB', (30, 20)).save(path)
        with open_image(path, draft_size=(5, 5)) as img:
            assert img.size == (30, 20)

    @requires_heic
    def test_heic_is_not_rotated_twice(self):
        """pillow-heif rotates while decoding and resets the tag, so the
        transpose in open_image must find nothing left to do."""
        with open_image(HEIC_PHOTO) as img:
            assert img.getexif().get(ExifTags.Base.Orientation, 1) == 1

    def test_preprocessed_tile_is_upright(self, temp_dir):
        """Tall tile from a photo stored wide but tagged portrait: the whole
        (rotated) image fits the tile, red on top."""
        path = os.path.join(temp_dir, 'phone.jpg')
        _save_oriented_jpeg(path, 6, size=(400, 200))
        config = PreprocessorConfig(target_width=20, target_height=40,
                                    enable_face_detection=False,
                                    enable_saliency=False,
                                    cache_dir=os.path.join(temp_dir, 'c'))
        tile = TilePreprocessor(config).preprocess_image(path)
        assert tile.size == (20, 40)
        assert _is_close(tile.getpixel((10, 5)), RED)
        assert _is_close(tile.getpixel((10, 35)), BLUE)

    def test_cache_key_includes_version(self, temp_dir, monkeypatch):
        """Entries cached before orientation was applied must not be
        reused, which relies on the version being part of the key."""
        path = os.path.join(temp_dir, 'a.png')
        Image.new('RGB', (10, 10)).save(path)
        cache = TilePreprocessorCache(os.path.join(temp_dir, 'c'))
        before = cache._get_cache_key(path, (10, 10))
        monkeypatch.setattr(tile_preprocessor, 'CACHE_VERSION',
                            tile_preprocessor.CACHE_VERSION + 1)
        assert cache._get_cache_key(path, (10, 10)) != before


class TestPipelineIntegration:

    @both_photos
    def test_tile_analyzer(self, path):
        tile = ImageTileAnalyzer().analyze_image(path)
        assert len(tile.sections) == 9

    @both_photos
    def test_preprocessor_produces_exact_tile(self, path, temp_dir):
        config = PreprocessorConfig(target_width=40, target_height=60,
                                    cache_dir=temp_dir)
        result = TilePreprocessor(config).preprocess_image(path)
        assert result.size == (40, 60)
        assert result.mode == 'RGB'

    @both_photos
    def test_guide_image(self, path):
        guide = GuideImage(path, output_width_px=400, output_height_px=300,
                           tile_width=40, tile_height=30)
        assert guide.grid_dimensions == (10, 10)

    def test_folder_scan_finds_uppercase_extensions(self, temp_dir):
        """iPhones write IMG_0001.HEIC; matching must ignore case."""
        for name in ('a.HEIC', 'b.heif', 'c.DNG', 'd.txt'):
            open(os.path.join(temp_dir, name), 'wb').close()
        found = TileDatabase().find_tile_files(temp_dir)
        assert sorted(os.path.basename(p) for p in found) == \
            ['a.HEIC', 'b.heif', 'c.DNG']

    @requires_heic
    @requires_dng
    def test_worker_processes_decode_both_formats(self, temp_dir):
        """Workers are fresh interpreters on Windows, so this proves the HEIF
        plugin registration and rawpy both reach them. Calls the pool
        directly: analyse_paths would fall back to serial and skip
        unreadable files, either of which could hide a worker failure."""
        paths = []
        for i in range(4):
            for src in (HEIC_PHOTO, DNG_PHOTO):
                dst = os.path.join(temp_dir, f'{i}_{os.path.basename(src)}')
                shutil.copy(src, dst)
                paths.append(dst)
        config = PreprocessorConfig(target_width=20, target_height=20,
                                    cache_dir=os.path.join(temp_dir, 'cache'))
        results = tile_loader._analyse_parallel(paths, config, 2, None, None)
        assert len(results) == len(paths)
        assert all(r is not None for r in results)
