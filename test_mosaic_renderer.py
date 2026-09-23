"""
Tests for the Mosaic Renderer Module
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
from PIL import Image

from guide_image import GuideImage
from tile_analyzer import ColorAverage, ImageTileAnalyzer
from tile_database import TileDatabase
from tile_preprocessor import PreprocessorConfig, TilePreprocessor
from mosaic_renderer import (
    MAX_REUSE_PENALTY,
    MIN_REUSE_PENALTY,
    MosaicRenderer,
    RenderConfig,
    RenderStats,
    TilePlacement,
    _TileBitmapCache,
    tint_tile,
)


TILE_W = 10
TILE_H = 10


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


def _write_tiles(folder, colors, size=(40, 40)):
    """Write one solid-colour image per colour and return their paths."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, color in enumerate(colors):
        path = os.path.join(folder, f"tile_{i:03d}.png")
        Image.new('RGB', size, color).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def tile_folder(temp_dir):
    """Eight strongly separated colours."""
    folder = os.path.join(temp_dir, 'tiles')
    _write_tiles(folder, [
        (0, 0, 0), (255, 255, 255),
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ])
    return folder


@pytest.fixture
def database(tile_folder):
    db = TileDatabase()
    db.load_tiles_from_folder(tile_folder)
    return db


@pytest.fixture
def guide_path(temp_dir):
    """A 4x3 cell guide: left half red, right half blue."""
    img = Image.new('RGB', (200, 150), (255, 0, 0))
    for x in range(100, 200):
        for y in range(150):
            img.putpixel((x, y), (0, 0, 255))
    path = os.path.join(temp_dir, 'guide.png')
    img.save(path)
    return path


@pytest.fixture
def guide(guide_path):
    # 40x30 px of grid at 10x10 tiles -> 4 cols x 3 rows
    return GuideImage(guide_path, 40, 30, TILE_W, TILE_H)


@pytest.fixture
def renderer(database):
    return MosaicRenderer(database, RenderConfig(TILE_W, TILE_H))


# ============================================================================
# RenderConfig
# ============================================================================

class TestRenderConfig:

    def test_defaults(self):
        config = RenderConfig()
        assert config.tile_size == (100, 100)
        assert config.max_tile_reuse == 0
        assert config.min_reuse_distance == 0
        assert config.variety == 0
        assert config.tint_strength == 0
        assert config.randomize_order is True

    def test_aspect_ratio(self):
        assert RenderConfig(120, 80).tile_aspect_ratio == pytest.approx(1.5)

    @pytest.mark.parametrize("kwargs", [
        {"tile_width": 0},
        {"tile_height": 0},
        {"tile_width": -5},
        {"candidate_pool": 0},
        {"max_tile_reuse": -1},
        {"min_reuse_distance": -1},
        {"tile_cache_mb": 0},
        {"tile_cache_size": 0},
        {"variety": -1},
        {"variety": 101},
        {"variety_pool": 0},
        {"tint_strength": -1},
        {"tint_strength": 101},
    ])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            RenderConfig(**kwargs)


class TestReusePenalty:
    """Variety maps onto the penalty logarithmically: the useful range
    spans three orders of magnitude."""

    def test_off_is_zero(self):
        assert RenderConfig(variety=0).reuse_penalty == 0

    def test_endpoints(self):
        assert RenderConfig(variety=1).reuse_penalty == pytest.approx(MIN_REUSE_PENALTY)
        assert RenderConfig(variety=100).reuse_penalty == pytest.approx(MAX_REUSE_PENALTY)

    def test_increases_with_variety(self):
        penalties = [RenderConfig(variety=v).reuse_penalty for v in range(101)]
        assert penalties == sorted(penalties)
        assert len(set(penalties)) == 101

    def test_midpoint_is_geometric(self):
        mid = RenderConfig(variety=50).reuse_penalty
        assert MIN_REUSE_PENALTY * 10 < mid < MAX_REUSE_PENALTY / 10


class TestTileCacheSizing:
    """The bitmap cache is budgeted in memory, not tile count: a count that
    is fine at 100x100 would be tens of gigabytes at 2000x2000."""

    def test_small_tiles_get_a_large_cache(self):
        # 256 MB / (100*100*3) is about 8,900 tiles.
        size = RenderConfig(100, 100).resolved_tile_cache_size()
        assert size > 1000

    def test_large_tiles_get_a_small_cache(self):
        size = RenderConfig(2000, 2000).resolved_tile_cache_size()
        assert size < 50

    def test_budget_is_respected(self):
        config = RenderConfig(100, 100, tile_cache_mb=8)
        bytes_used = config.resolved_tile_cache_size() * 100 * 100 * 3
        assert bytes_used <= 8 * 1024 * 1024

    def test_explicit_override_wins(self):
        assert RenderConfig(100, 100, tile_cache_size=7
                            ).resolved_tile_cache_size() == 7

    def test_never_zero(self):
        assert RenderConfig(2000, 2000, tile_cache_mb=1
                            ).resolved_tile_cache_size() >= 1

    def test_renderer_uses_the_resolved_size(self, database):
        config = RenderConfig(TILE_W, TILE_H, tile_cache_size=3)
        renderer = MosaicRenderer(database, config)
        assert renderer._cache._maxsize == 3


# ============================================================================
# Planning
# ============================================================================

class TestPlan:

    def test_one_placement_per_cell(self, renderer, guide):
        placements = renderer.plan(guide)
        rows, cols = guide.grid_dimensions
        assert len(placements) == rows * cols

    def test_covers_every_cell_exactly_once(self, renderer, guide):
        placements = renderer.plan(guide)
        rows, cols = guide.grid_dimensions
        seen = {(p.row, p.col) for p in placements}
        assert seen == {(r, c) for r in range(rows) for c in range(cols)}

    def test_row_major_order(self, renderer, guide):
        placements = renderer.plan(guide)
        rows, cols = guide.grid_dimensions
        expected = [(r, c) for r in range(rows) for c in range(cols)]
        assert [(p.row, p.col) for p in placements] == expected

    def test_matches_colours_sensibly(self, renderer, guide):
        """Red cells should get the red tile, blue cells the blue tile."""
        placements = {(p.row, p.col): p for p in renderer.plan(guide)}

        left = placements[(0, 0)]
        right = placements[(0, 3)]

        def dominant(placement):
            colors = placement.tile_data.get_all_colors()
            avg = tuple(round(sum(getattr(c, ch) for c in colors) / len(colors))
                        for ch in ('r', 'g', 'b'))
            return avg

        assert dominant(left)[0] > dominant(left)[2], "left half should be reddish"
        assert dominant(right)[2] > dominant(right)[0], "right half should be bluish"

    def test_row_major_order_when_randomized(self, database, guide):
        """Shuffling changes the visiting order, not the returned order."""
        renderer = MosaicRenderer(
            database, RenderConfig(TILE_W, TILE_H, randomize_order=True))
        rows, cols = guide.grid_dimensions
        expected = [(r, c) for r in range(rows) for c in range(cols)]
        assert [(p.row, p.col) for p in renderer.plan(guide)] == expected

    def test_target_colours_are_recorded(self, renderer, guide):
        for p in renderer.plan(guide):
            expected = [c.as_tuple() for c in guide.get_cell_colors(p.row, p.col)]
            assert list(p.target_colors) == expected

    def test_distances_are_recorded(self, renderer, guide):
        assert all(p.distance >= 0 for p in renderer.plan(guide))

    def test_unconstrained_plan_is_not_marked_relaxed(self, renderer, guide):
        assert all(not p.constraint_relaxed for p in renderer.plan(guide))

    def test_progress_callback_reports_completion(self, renderer, guide):
        calls = []
        renderer.plan(guide, progress_callback=lambda c, t, m: calls.append((c, t, m)))

        rows, cols = guide.grid_dimensions
        total = rows * cols
        assert calls, "progress callback was never invoked"
        assert calls[-1][0] == total
        assert all(t == total for _, t, _ in calls)
        assert all(1 <= c <= total for c, _, _ in calls)

    def test_rejects_empty_database(self, guide):
        renderer = MosaicRenderer(TileDatabase(), RenderConfig(TILE_W, TILE_H))
        with pytest.raises(RuntimeError, match="empty"):
            renderer.plan(guide)

    def test_rejects_unbuilt_index(self, tile_folder, guide):
        db = TileDatabase()
        db.load_tiles_from_folder(tile_folder, auto_build_index=False)
        renderer = MosaicRenderer(db, RenderConfig(TILE_W, TILE_H))
        with pytest.raises(RuntimeError, match="index"):
            renderer.plan(guide)


# ============================================================================
# Repetition constraints
# ============================================================================

class TestRepetitionConstraints:

    def _uniform_guide(self, temp_dir, cols=6, rows=6):
        """A flat grey guide - the worst case for tile repetition."""
        img = Image.new('RGB', (cols * TILE_W, rows * TILE_H), (128, 128, 128))
        path = os.path.join(temp_dir, 'flat.png')
        img.save(path)
        return GuideImage(path, cols * TILE_W, rows * TILE_H, TILE_W, TILE_H)

    def test_flat_guide_repeats_one_tile_without_constraints(self, database, temp_dir):
        """Establishes the problem the constraints exist to solve."""
        guide = self._uniform_guide(temp_dir)
        renderer = MosaicRenderer(database, RenderConfig(TILE_W, TILE_H))
        stats = renderer.summarize(renderer.plan(guide))
        assert stats.distinct_tiles == 1
        assert stats.max_tile_uses == stats.total_cells

    def test_reuse_cap_increases_variety(self, database, temp_dir):
        guide = self._uniform_guide(temp_dir)
        unconstrained = MosaicRenderer(database, RenderConfig(TILE_W, TILE_H))
        capped = MosaicRenderer(
            database,
            RenderConfig(TILE_W, TILE_H, max_tile_reuse=5, candidate_pool=8)
        )

        before = unconstrained.summarize(unconstrained.plan(guide))
        after = capped.summarize(capped.plan(guide))

        assert after.distinct_tiles > before.distinct_tiles
        assert after.max_tile_uses < before.max_tile_uses

    def test_reuse_cap_is_honoured_when_achievable(self, database, temp_dir):
        """8 tiles x 5 uses = 40 placements for 36 cells, so the cap holds."""
        guide = self._uniform_guide(temp_dir)
        renderer = MosaicRenderer(
            database,
            RenderConfig(TILE_W, TILE_H, max_tile_reuse=5, candidate_pool=8)
        )
        stats = renderer.summarize(renderer.plan(guide))
        assert stats.max_tile_uses <= 5

    def test_adjacency_constraint_separates_repeats(self, database, temp_dir):
        """No tile may repeat within the configured Chebyshev radius."""
        guide = self._uniform_guide(temp_dir)
        distance = 1
        renderer = MosaicRenderer(
            database,
            RenderConfig(TILE_W, TILE_H, min_reuse_distance=distance,
                         candidate_pool=8)
        )
        placements = renderer.plan(guide)

        grid = {(p.row, p.col): p.image_path for p in placements}
        violations = []
        for (row, col), path in grid.items():
            for dr in range(-distance, distance + 1):
                for dc in range(-distance, distance + 1):
                    if dr == 0 and dc == 0:
                        continue
                    if grid.get((row + dr, col + dc)) == path:
                        violations.append(((row, col), (row + dr, col + dc)))

        assert not violations, f"adjacent repeats found: {violations[:5]}"

    def test_relaxes_rather_than_failing(self, temp_dir):
        """An impossible cap must still produce a complete mosaic."""
        folder = os.path.join(temp_dir, 'one')
        _write_tiles(folder, [(10, 10, 10)])
        db = TileDatabase()
        db.load_tiles_from_folder(folder)

        guide = self._uniform_guide(temp_dir, cols=4, rows=4)
        renderer = MosaicRenderer(
            db, RenderConfig(TILE_W, TILE_H, max_tile_reuse=1)
        )
        placements = renderer.plan(guide)

        assert len(placements) == 16
        assert any(p.constraint_relaxed for p in placements)

    def test_relaxation_prefers_the_least_used_candidate(self, database, temp_dir):
        """When the pool is exhausted, fall back to the least-used tile.

        Falling back to the nearest match instead would keep piling cells
        onto whichever tile is already the most overused.
        """
        import math

        guide = self._uniform_guide(temp_dir, cols=8, rows=8)
        renderer = MosaicRenderer(
            database,
            RenderConfig(TILE_W, TILE_H, max_tile_reuse=2, candidate_pool=3)
        )
        placements = renderer.plan(guide)
        stats = renderer.summarize(placements)

        # The pool caps how many tiles are reachable, so an even split over
        # the tiles actually used is the best achievable outcome. Falling
        # back to the nearest match instead would put most of the 64 cells
        # on a single tile.
        even_split = math.ceil(stats.total_cells / stats.distinct_tiles)
        assert stats.max_tile_uses <= even_split + 2, stats

    def test_adjacency_holds_in_every_direction_when_shuffled(self, database, temp_dir):
        """With a shuffled order, neighbours below and to the right can be
        placed first, so the check must look all the way round."""
        guide = self._uniform_guide(temp_dir, cols=8, rows=8)
        for seed in range(5):
            renderer = MosaicRenderer(database, RenderConfig(
                TILE_W, TILE_H, min_reuse_distance=1, candidate_pool=8,
                randomize_order=True, seed=seed))
            grid = {(p.row, p.col): p.image_path for p in renderer.plan(guide)}
            for (row, col), path in grid.items():
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (dr or dc) and grid.get((row + dr, col + dc)) == path:
                            pytest.fail(f"seed {seed}: repeat at {(row, col)}")

    def test_impossible_cap_is_logged(self, database, temp_dir, caplog):
        guide = self._uniform_guide(temp_dir, cols=10, rows=10)
        renderer = MosaicRenderer(
            database, RenderConfig(TILE_W, TILE_H, max_tile_reuse=2)
        )
        with caplog.at_level('WARNING'):
            renderer.plan(guide)
        assert any('max_tile_reuse' in r.message for r in caplog.records)


# ============================================================================
# Visiting order
# ============================================================================

class TestVisitingOrder:

    def _flat_guide(self, temp_dir, cols=8, rows=8):
        img = Image.new('RGB', (cols * TILE_W, rows * TILE_H), (128, 128, 128))
        path = os.path.join(temp_dir, 'flat.png')
        img.save(path)
        return GuideImage(path, cols * TILE_W, rows * TILE_H, TILE_W, TILE_H)

    def _capped(self, database, **kw):
        # 8 tiles x 4 uses = 32 placements for 64 cells: half must relax.
        return MosaicRenderer(database, RenderConfig(
            TILE_W, TILE_H, max_tile_reuse=4, candidate_pool=8, **kw))

    def test_row_major_leaves_the_bottom_with_the_leftovers(self, database, temp_dir):
        """Establishes the problem shuffling solves."""
        guide = self._flat_guide(temp_dir)
        placements = self._capped(database, randomize_order=False).plan(guide)
        relaxed_rows = {p.row for p in placements if p.constraint_relaxed}
        assert min(relaxed_rows) >= 4, "top half should get every good tile"

    def test_shuffled_spreads_the_compromise(self, database, temp_dir):
        guide = self._flat_guide(temp_dir)
        placements = self._capped(database, randomize_order=True).plan(guide)
        top = sum(p.constraint_relaxed for p in placements if p.row < 4)
        bottom = sum(p.constraint_relaxed for p in placements if p.row >= 4)
        assert top > 0 and bottom > 0
        assert abs(top - bottom) <= 12

    def test_shuffle_is_reproducible(self, database, temp_dir):
        guide = self._flat_guide(temp_dir)
        first = self._capped(database, seed=7).plan(guide)
        second = self._capped(database, seed=7).plan(guide)
        assert [p.image_path for p in first] == [p.image_path for p in second]

    def test_seed_changes_the_layout(self, database, temp_dir):
        guide = self._flat_guide(temp_dir)
        first = self._capped(database, seed=1).plan(guide)
        second = self._capped(database, seed=2).plan(guide)
        assert [p.image_path for p in first] != [p.image_path for p in second]


# ============================================================================
# Variety (soft reuse penalty)
# ============================================================================

class TestVariety:

    @pytest.fixture
    def grey_library(self, temp_dir):
        """Twelve greys, one level apart, moving away from the guide's 128.

        Close together, like the near-equivalent photos a real library
        has for any flat area, so the small penalty can reach them.
        """
        folder = os.path.join(temp_dir, 'greys')
        _write_tiles(folder, [(128 + i,) * 3 for i in range(12)])
        db = TileDatabase()
        db.load_tiles_from_folder(folder)
        return db

    def _flat_guide(self, temp_dir):
        img = Image.new('RGB', (10 * TILE_W, 10 * TILE_H), (128, 128, 128))
        path = os.path.join(temp_dir, 'flat.png')
        img.save(path)
        return GuideImage(path, 10 * TILE_W, 10 * TILE_H, TILE_W, TILE_H)

    def _stats(self, db, guide, **kw):
        renderer = MosaicRenderer(db, RenderConfig(TILE_W, TILE_H, **kw))
        return renderer.summarize(renderer.plan(guide))

    def test_off_uses_only_the_best_tile(self, grey_library, temp_dir):
        stats = self._stats(grey_library, self._flat_guide(temp_dir), variety=0)
        assert stats.distinct_tiles == 1

    def test_uses_more_tiles(self, grey_library, temp_dir):
        stats = self._stats(grey_library, self._flat_guide(temp_dir), variety=100)
        assert stats.distinct_tiles > 1
        assert stats.max_tile_uses < stats.total_cells

    def test_more_variety_never_uses_fewer_tiles(self, grey_library, temp_dir):
        guide = self._flat_guide(temp_dir)
        counts = [self._stats(grey_library, guide, variety=v).distinct_tiles
                  for v in (0, 25, 50, 75, 100)]
        assert counts == sorted(counts), counts
        assert counts[-1] > counts[0]

    def test_trades_colour_accuracy_for_variety(self, grey_library, temp_dir):
        guide = self._flat_guide(temp_dir)
        off = self._stats(grey_library, guide, variety=0)
        on = self._stats(grey_library, guide, variety=100)
        assert on.mean_distance > off.mean_distance

    def test_does_not_override_a_far_better_match(self, database, guide):
        """A small penalty must not put a green tile on a red cell."""
        renderer = MosaicRenderer(database, RenderConfig(
            TILE_W, TILE_H, variety=100))
        for p in renderer.plan(guide):
            avg = np.mean([c.as_tuple() for c in p.tile_data.get_all_colors()], axis=0)
            if p.col < 2:
                assert avg[0] > avg[2], p
            else:
                assert avg[2] > avg[0], p

    def test_hard_cap_still_holds(self, grey_library, temp_dir):
        stats = self._stats(grey_library, self._flat_guide(temp_dir),
                            variety=30, max_tile_reuse=10)
        assert stats.max_tile_uses <= 10

    def test_widens_the_candidate_pool(self, grey_library, temp_dir):
        """Only tiles in the shortlist can be reached, so variety must look
        further than the hard-limit pool does."""
        guide = self._flat_guide(temp_dir)
        narrow = self._stats(grey_library, guide, variety=100, variety_pool=2)
        wide = self._stats(grey_library, guide, variety=100, variety_pool=12)
        assert narrow.distinct_tiles <= 2
        assert wide.distinct_tiles > narrow.distinct_tiles


# ============================================================================
# Colour tint
# ============================================================================

def _striped(size=12, low=60, high=120):
    """A tile of alternating light and dark columns, averaging 90."""
    pixels = np.zeros((size, size, 3), np.uint8)
    pixels[:, ::2] = high
    pixels[:, 1::2] = low
    return Image.fromarray(pixels)


class TestTintTile:

    def test_zero_strength_is_a_no_op(self):
        tile = _striped()
        assert tint_tile(tile, [(255, 0, 0)] * 9, 0) is tile

    def test_full_strength_reaches_the_target(self):
        tile = Image.new('RGB', (12, 12), (40, 40, 40))
        tinted = np.asarray(tint_tile(tile, [(200, 100, 50)] * 9, 1.0))
        assert np.allclose(tinted, (200, 100, 50), atol=1)

    def test_half_strength_goes_halfway(self):
        tile = Image.new('RGB', (12, 12), (40, 40, 40))
        tinted = np.asarray(tint_tile(tile, [(200, 100, 40)] * 9, 0.5))
        assert np.allclose(tinted, (120, 70, 40), atol=1)

    def test_keeps_the_tile_detail(self):
        """Additive, not a blend: the stripes keep their contrast."""
        tile = _striped()
        tinted = np.asarray(tint_tile(tile, [(150, 150, 150)] * 9, 1.0), float)
        original = np.asarray(tile, float)
        assert tinted.mean() == pytest.approx(150, abs=1)
        assert tinted.std() == pytest.approx(original.std(), abs=1)

    def test_follows_each_section(self):
        """Each third of the tile moves toward its own target."""
        tile = Image.new('RGB', (30, 30), (100, 100, 100))
        targets = [(250, 0, 0), (250, 0, 0), (250, 0, 0),
                   (100, 100, 100), (100, 100, 100), (100, 100, 100),
                   (0, 0, 250), (0, 0, 250), (0, 0, 250)]
        tinted = np.asarray(tint_tile(tile, targets, 1.0))
        top, bottom = tinted[2, 15], tinted[27, 15]
        assert top[0] > 200 and top[2] < 50
        assert bottom[2] > 200 and bottom[0] < 50

    def test_clips_instead_of_wrapping(self):
        tile = _striped(low=200, high=255)
        tinted = np.asarray(tint_tile(tile, [(255, 255, 255)] * 9, 1.0))
        assert tinted.min() >= 200   # uint8 wraparound would show as dark

    def test_preserves_size_and_mode(self):
        tile = Image.new('RGB', (17, 9), (10, 20, 30))
        tinted = tint_tile(tile, [(90, 90, 90)] * 9, 0.4)
        assert tinted.size == (17, 9)
        assert tinted.mode == 'RGB'


# ============================================================================
# Compositing
# ============================================================================

class TestComposite:

    def test_canvas_size_matches_grid(self, renderer, guide):
        placements = renderer.plan(guide)
        image = renderer.composite(placements, guide.grid_dimensions)
        rows, cols = guide.grid_dimensions
        assert image.size == (cols * TILE_W, rows * TILE_H)

    def test_output_is_rgb(self, renderer, guide):
        image = renderer.composite(renderer.plan(guide), guide.grid_dimensions)
        assert image.mode == 'RGB'

    def test_tiles_land_in_the_right_cells(self, renderer, guide):
        """The left half should read red, the right half blue."""
        image = renderer.composite(renderer.plan(guide), guide.grid_dimensions)
        left = image.getpixel((TILE_W // 2, TILE_H // 2))
        right = image.getpixel((3 * TILE_W + TILE_W // 2, TILE_H // 2))
        assert left[0] > left[2]
        assert right[2] > right[0]

    def test_tint_pulls_tiles_toward_the_guide(self, temp_dir, guide):
        """A lone black tile, fully tinted, reproduces the red/blue guide."""
        folder = os.path.join(temp_dir, 'black')
        _write_tiles(folder, [(0, 0, 0)])
        db = TileDatabase()
        db.load_tiles_from_folder(folder)
        renderer = MosaicRenderer(db, RenderConfig(TILE_W, TILE_H,
                                                   tint_strength=100))
        image = renderer.composite(renderer.plan(guide), guide.grid_dimensions)
        assert image.getpixel((5, 5))[0] > 200
        assert image.getpixel((35, 5))[2] > 200

    def test_no_tint_leaves_tiles_untouched(self, temp_dir, guide):
        folder = os.path.join(temp_dir, 'black')
        _write_tiles(folder, [(0, 0, 0)])
        db = TileDatabase()
        db.load_tiles_from_folder(folder)
        renderer = MosaicRenderer(db, RenderConfig(TILE_W, TILE_H))
        image = renderer.composite(renderer.plan(guide), guide.grid_dimensions)
        assert image.getpixel((5, 5)) == (0, 0, 0)

    def test_tint_does_not_alter_the_cached_tile(self, temp_dir, guide):
        """One tile placed on red and blue cells must not carry one cell's
        tint into the next."""
        folder = os.path.join(temp_dir, 'black')
        paths = _write_tiles(folder, [(0, 0, 0)])
        db = TileDatabase()
        db.load_tiles_from_folder(folder)
        renderer = MosaicRenderer(db, RenderConfig(TILE_W, TILE_H,
                                                   tint_strength=100))
        renderer.composite(renderer.plan(guide), guide.grid_dimensions)
        cached = renderer._cache.get(paths[0])
        assert cached.getpixel((5, 5)) == (0, 0, 0)

    def test_rejects_empty_grid(self, renderer):
        with pytest.raises(ValueError):
            renderer.composite([], (0, 0))

    def test_unreadable_tile_does_not_abort_render(self, renderer, guide, temp_dir):
        """A missing tile file should leave one cell black, not crash."""
        placements = renderer.plan(guide)
        ghost = TilePlacement(
            row=0, col=0,
            tile_data=type(placements[0].tile_data)(
                os.path.join(temp_dir, 'does_not_exist.png'),
                placements[0].tile_data.sections,
            ),
            distance=0.0,
        )
        image = renderer.composite([ghost] + placements[1:],
                                   guide.grid_dimensions)
        assert image.getpixel((0, 0)) == (0, 0, 0)

    def test_progress_callback_reports_completion(self, renderer, guide):
        placements = renderer.plan(guide)
        calls = []
        renderer.composite(placements, guide.grid_dimensions,
                           progress_callback=lambda c, t, m: calls.append((c, t, m)))
        assert calls[-1][0] == len(placements)


# ============================================================================
# Tile preparation
# ============================================================================

class TestTilePreparation:

    def test_tiles_are_resized_to_exact_tile_size(self, database, guide):
        """The preprocessor only crops to an aspect ratio, so the renderer
        is responsible for producing exactly tile_width x tile_height."""
        renderer = MosaicRenderer(database, RenderConfig(TILE_W, TILE_H))
        path = database.get_tile_by_index(0).image_path
        assert renderer._get_tile_bitmap(path).size == (TILE_W, TILE_H)

    def test_non_square_tiles(self, database, temp_dir):
        renderer = MosaicRenderer(database, RenderConfig(24, 12))
        path = database.get_tile_by_index(0).image_path
        assert renderer._get_tile_bitmap(path).size == (24, 12)

    def test_preprocessor_already_returns_tile_size(self, tile_folder, temp_dir):
        """The preprocessor now resizes to the configured tile size itself."""
        config = PreprocessorConfig(target_width=TILE_W, target_height=TILE_H,
                                    enable_face_detection=False,
                                    enable_saliency=False)
        preprocessor = TilePreprocessor(config)
        db = TileDatabase(preprocessor=preprocessor)
        db.load_tiles_from_folder(tile_folder)

        renderer = MosaicRenderer(db, RenderConfig(TILE_W, TILE_H),
                                  preprocessor=preprocessor)
        path = db.get_tile_by_index(0).image_path

        assert preprocessor.preprocess_image(path).size == (TILE_W, TILE_H)
        assert renderer._get_tile_bitmap(path).size == (TILE_W, TILE_H)

    def test_renderer_resizes_a_mismatched_preprocessor(self, tile_folder):
        """The renderer must still guarantee the tile size it was asked for
        even if handed a preprocessor configured for different dimensions."""
        preprocessor = TilePreprocessor(PreprocessorConfig(
            target_width=32, target_height=32,
            enable_face_detection=False, enable_saliency=False))
        db = TileDatabase(preprocessor=preprocessor)
        db.load_tiles_from_folder(tile_folder)

        renderer = MosaicRenderer(db, RenderConfig(TILE_W, TILE_H),
                                  preprocessor=preprocessor)
        path = db.get_tile_by_index(0).image_path

        assert preprocessor.preprocess_image(path).size == (32, 32)
        assert renderer._get_tile_bitmap(path).size == (TILE_W, TILE_H)

    def test_missing_file_returns_none(self, renderer, temp_dir):
        assert renderer._get_tile_bitmap(os.path.join(temp_dir, 'nope.png')) is None


# ============================================================================
# Bitmap cache
# ============================================================================

class TestTileBitmapCache:

    def test_stores_and_returns(self):
        cache = _TileBitmapCache(4)
        img = Image.new('RGB', (4, 4))
        cache.put('a', img)
        assert cache.get('a') is img

    def test_miss_returns_none(self):
        assert _TileBitmapCache(4).get('absent') is None

    def test_evicts_least_recently_used(self):
        cache = _TileBitmapCache(2)
        for key in ('a', 'b'):
            cache.put(key, Image.new('RGB', (2, 2)))
        cache.get('a')                      # 'b' is now least recent
        cache.put('c', Image.new('RGB', (2, 2)))

        assert cache.get('a') is not None
        assert cache.get('b') is None
        assert cache.get('c') is not None

    def test_never_exceeds_maxsize(self):
        cache = _TileBitmapCache(3)
        for i in range(20):
            cache.put(str(i), Image.new('RGB', (2, 2)))
        assert len(cache) == 3

    def test_repeated_tiles_hit_the_cache(self, database, temp_dir):
        """A flat guide reuses tiles heavily; that must not re-decode."""
        img = Image.new('RGB', (6 * TILE_W, 6 * TILE_H), (128, 128, 128))
        path = os.path.join(temp_dir, 'flat.png')
        img.save(path)
        guide = GuideImage(path, 6 * TILE_W, 6 * TILE_H, TILE_W, TILE_H)

        renderer = MosaicRenderer(database, RenderConfig(TILE_W, TILE_H))
        renderer.render(guide)
        assert renderer._cache.hits > 0


# ============================================================================
# render() and summarize()
# ============================================================================

class TestRender:

    def test_render_matches_plan_then_composite(self, database, guide):
        a = MosaicRenderer(database, RenderConfig(TILE_W, TILE_H))
        b = MosaicRenderer(database, RenderConfig(TILE_W, TILE_H))

        direct = a.render(guide)
        staged = b.composite(b.plan(guide), guide.grid_dimensions)

        assert direct.size == staged.size
        assert list(direct.getdata()) == list(staged.getdata())

    def test_render_reports_both_phases(self, renderer, guide):
        messages = set()
        renderer.render(guide, progress_callback=lambda c, t, m: messages.add(m))
        assert len(messages) == 2, messages


class TestSummarize:

    def test_empty_plan(self):
        stats = MosaicRenderer.summarize([])
        assert stats == RenderStats()

    def test_counts(self, renderer, guide):
        placements = renderer.plan(guide)
        stats = renderer.summarize(placements)

        rows, cols = guide.grid_dimensions
        distinct = len({p.image_path for p in placements})

        assert stats.total_cells == rows * cols
        assert stats.distinct_tiles == distinct
        assert stats.max_tile_uses >= 1
        assert stats.mean_distance >= 0
