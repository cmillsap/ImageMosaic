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
from PIL import Image

from guide_image import GuideImage
from image_io import open_image
from tile_analyzer import ImageTileAnalyzer
from tile_database import TileDatabase
import tile_loader
from tile_preprocessor import PreprocessorConfig, TilePreprocessor

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
