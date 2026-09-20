"""
Tests for the Parallel Tile Loader.

Most cases stay under PARALLEL_THRESHOLD so they exercise the serial path
quickly; a couple cross it deliberately to prove the process pool works,
since that is where pickling and worker initialisation can break.
"""

import os
import shutil
import tempfile

import pytest
from PIL import Image

import tile_loader
from tile_analyzer import TileData
from tile_database import DEFAULT_IMAGE_EXTENSIONS, TileDatabase
from tile_loader import (
    DEFAULT_MAX_WORKERS,
    PARALLEL_THRESHOLD,
    analyse_paths,
    resolve_worker_count,
)
from tile_preprocessor import PreprocessorConfig


TILE = 10


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


def make_tiles(folder, count, size=(30, 30)):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i in range(count):
        path = os.path.join(folder, f"t{i:04d}.png")
        Image.new('RGB', size, (i % 256, (i * 7) % 256, (i * 13) % 256)).save(path)
        paths.append(path)
    return paths


# ============================================================================
# Worker count
# ============================================================================

class TestWorkerCount:

    def test_explicit_value_wins(self):
        assert resolve_worker_count(3) == 3

    def test_default_is_capped(self):
        assert 1 <= resolve_worker_count(None) <= DEFAULT_MAX_WORKERS

    def test_zero_and_negative_fall_back_to_default(self):
        assert resolve_worker_count(0) == resolve_worker_count(None)
        assert resolve_worker_count(-4) == resolve_worker_count(None)


# ============================================================================
# Serial path
# ============================================================================

class TestSerialPath:

    def test_empty_input(self, config):
        assert analyse_paths([], config) == []

    def test_analyses_every_file(self, temp_dir, config):
        paths = make_tiles(os.path.join(temp_dir, 't'), 5)
        tiles = analyse_paths(paths, config, max_workers=1)

        assert len(tiles) == 5
        assert all(isinstance(t, TileData) for t in tiles)

    def test_results_carry_source_paths(self, temp_dir, config):
        paths = make_tiles(os.path.join(temp_dir, 't'), 4)
        tiles = analyse_paths(paths, config, max_workers=1)
        assert {t.image_path for t in tiles} == set(paths)

    def test_unreadable_files_are_skipped_not_fatal(self, temp_dir, config):
        """One corrupt file must not abort a long run."""
        paths = make_tiles(os.path.join(temp_dir, 't'), 4)
        broken = os.path.join(temp_dir, 't', 'broken.png')
        with open(broken, 'wb') as handle:
            handle.write(b'not an image at all')

        tiles = analyse_paths(paths + [broken], config, max_workers=1)
        assert len(tiles) == 4

    def test_progress_reaches_the_total(self, temp_dir, config):
        paths = make_tiles(os.path.join(temp_dir, 't'), 6)
        calls = []
        analyse_paths(paths, config, max_workers=1,
                      progress_callback=lambda c, t, m: calls.append((c, t)))

        assert calls[-1] == (6, 6)
        assert all(t == 6 for _, t in calls)

    def test_cancellation_stops_early(self, temp_dir, config):
        paths = make_tiles(os.path.join(temp_dir, 't'), 20)
        seen = []

        def cancel_after_three():
            seen.append(1)
            return len(seen) > 3

        tiles = analyse_paths(paths, config, max_workers=1,
                              should_cancel=cancel_after_three)
        assert len(tiles) < 20


# ============================================================================
# Threshold
# ============================================================================

class TestParallelThreshold:

    def test_small_batches_stay_serial(self, temp_dir, config, monkeypatch):
        """Starting a pool costs ~0.8s, so tiny batches must not pay it."""
        called = []
        monkeypatch.setattr(tile_loader, '_analyse_parallel',
                            lambda *a, **k: called.append(1) or [])

        paths = make_tiles(os.path.join(temp_dir, 't'), PARALLEL_THRESHOLD - 1)
        analyse_paths(paths, config)
        assert called == []

    def test_large_batches_go_parallel(self, temp_dir, config, monkeypatch):
        called = []

        def fake(paths, cfg, workers, progress, cancel):
            called.append(workers)
            return []

        monkeypatch.setattr(tile_loader, '_analyse_parallel', fake)
        paths = make_tiles(os.path.join(temp_dir, 't'), PARALLEL_THRESHOLD + 1)
        analyse_paths(paths, config)
        assert len(called) == 1


# ============================================================================
# Process pool
# ============================================================================

class TestProcessPool:
    """Crosses the threshold for real: pickling and worker init break here."""

    def test_parallel_matches_serial(self, temp_dir, config):
        paths = make_tiles(os.path.join(temp_dir, 't'), PARALLEL_THRESHOLD + 8)

        serial = analyse_paths(paths, config, max_workers=1)
        parallel = analyse_paths(paths, config, max_workers=2)

        assert len(parallel) == len(serial)

        def signature(tiles):
            return {t.image_path: [c.as_tuple() for c in t.get_all_colors()]
                    for t in tiles}

        assert signature(parallel) == signature(serial)

    def test_falls_back_when_the_pool_cannot_start(self, temp_dir, config, monkeypatch):
        """Slow beats broken: a frozen build or locked-down sandbox."""
        def explode(*args, **kwargs):
            raise OSError("no process spawning here")

        monkeypatch.setattr(tile_loader, '_analyse_parallel', explode)
        paths = make_tiles(os.path.join(temp_dir, 't'), PARALLEL_THRESHOLD + 2)

        tiles = analyse_paths(paths, config)
        assert len(tiles) == len(paths)


# ============================================================================
# Extension coverage
# ============================================================================

class TestDefaultExtensions:

    def test_includes_tiff(self):
        """41 TIFFs in a real library were being skipped; Pillow reads them."""
        assert '.tif' in DEFAULT_IMAGE_EXTENSIONS
        assert '.tiff' in DEFAULT_IMAGE_EXTENSIONS

    def test_includes_cr2(self):
        """Pillow opens Canon .cr2 directly, at full resolution."""
        assert '.cr2' in DEFAULT_IMAGE_EXTENSIONS

    @pytest.mark.parametrize("ext", ['.crw', '.cr3', '.psd'])
    def test_excludes_formats_pillow_cannot_read(self, ext):
        assert ext not in DEFAULT_IMAGE_EXTENSIONS

    def test_excludes_dng(self):
        """Pillow opens .dng but returns only a ~256px embedded thumbnail,
        which is too small for face detection to be meaningful."""
        assert '.dng' not in DEFAULT_IMAGE_EXTENSIONS

    def test_tiff_files_are_discovered(self, temp_dir):
        folder = os.path.join(temp_dir, 'mixed')
        os.makedirs(folder)
        Image.new('RGB', (20, 20), (1, 2, 3)).save(os.path.join(folder, 'a.tif'))
        Image.new('RGB', (20, 20), (4, 5, 6)).save(os.path.join(folder, 'b.png'))

        found = TileDatabase().find_tile_files(folder)
        assert len(found) == 2
        assert any(p.endswith('.tif') for p in found)


# ============================================================================
# Database integration
# ============================================================================

class TestDatabaseIntegration:

    def test_load_tiles_parallel_populates_and_indexes(self, temp_dir, config):
        folder = os.path.join(temp_dir, 'tiles')
        make_tiles(folder, 12)

        db = TileDatabase()
        count = db.load_tiles_parallel(folder, config, max_workers=1)

        assert count == 12
        assert db.size() == 12
        assert db.is_ready() is True

    def test_recursive_scan(self, temp_dir, config):
        folder = os.path.join(temp_dir, 'tiles')
        make_tiles(folder, 4)
        make_tiles(os.path.join(folder, 'nested'), 3)

        db = TileDatabase()
        assert db.load_tiles_parallel(folder, config, max_workers=1,
                                      recursive=True) == 7
        db2 = TileDatabase()
        assert db2.load_tiles_parallel(folder, config, max_workers=1,
                                       recursive=False) == 4

    def test_cancelled_load_does_not_build_index(self, temp_dir, config):
        folder = os.path.join(temp_dir, 'tiles')
        make_tiles(folder, 10)

        db = TileDatabase()
        db.load_tiles_parallel(folder, config, max_workers=1,
                               should_cancel=lambda: True)
        assert db.is_ready() is False

    def test_missing_folder_raises(self, temp_dir, config):
        with pytest.raises(FileNotFoundError):
            TileDatabase().load_tiles_parallel(
                os.path.join(temp_dir, 'absent'), config)

    def test_matches_the_serial_loader(self, temp_dir, config):
        """The two entry points must agree on what they produce."""
        from tile_preprocessor import TilePreprocessor

        folder = os.path.join(temp_dir, 'tiles')
        make_tiles(folder, 8)

        serial_db = TileDatabase(preprocessor=TilePreprocessor(config))
        serial_db.load_tiles_from_folder(folder)

        parallel_db = TileDatabase()
        parallel_db.load_tiles_parallel(folder, config, max_workers=1)

        def signature(db):
            return sorted(
                (os.path.basename(t.image_path),
                 [c.as_tuple() for c in t.get_all_colors()])
                for t in db.get_all_tiles()
            )

        assert signature(parallel_db) == signature(serial_db)
