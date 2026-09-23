"""
Mosaic Renderer Module

Bridges the guide image grid and the tile database: decides which tile goes
in each grid cell, then composites the chosen tiles into the final image.

Rendering is split into two phases so the matching logic can be tested and
reported on without touching pixels:

    plan()      GuideImage + TileDatabase -> list[TilePlacement]
    composite() list[TilePlacement]       -> PIL Image

render() runs both.
"""

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from guide_image import GuideImage
from image_io import open_image
from tile_analyzer import TileData
from tile_database import TileDatabase
from tile_preprocessor import CropCalculator, TilePreprocessor

logger = logging.getLogger(__name__)

# Called as callback(completed, total, message).
ProgressCallback = Callable[[int, int, str], None]
# Polled during long loops; returning True aborts with RenderCancelled.
CancelCheck = Callable[[], bool]


# Colour-distance penalty per earlier use of a tile, at variety 1 and 100.
# Variety maps onto this range logarithmically, because the useful range
# spans three orders of magnitude: flat areas of a guide have many tiles
# only slightly worse than the best, so even a tiny penalty moves cells onto
# them once the best tile has been used hundreds of times. Measured on 600
# tiles / 5,400 cells (colour distance is Euclidean over the 27 values):
#     penalty 0     ->  44 distinct, most-used tile 3224x, mean distance 131
#     penalty 0.05  ->  52 distinct, most-used tile 2120x, mean distance 143
#     penalty 1     -> 148 distinct, most-used tile  270x, mean distance 217
#     penalty 50    -> 389 distinct, most-used tile   34x, mean distance 300
# Past 50 nothing changes: every tile in reach is already in use.
MIN_REUSE_PENALTY = 0.05
MAX_REUSE_PENALTY = 50.0


class RenderCancelled(Exception):
    """Raised when a caller's should_cancel check aborts a render."""


@dataclass
class RenderConfig:
    """Configuration for a mosaic render.

    Tile dimensions must match the ones the GuideImage was built with, or
    cells and tiles will disagree in shape.
    """

    tile_width: int = 100
    tile_height: int = 100

    # Repetition control. A plain nearest-neighbour match reuses the same
    # photo across any large flat area (a sky becomes one image 500 times),
    # so candidates can be rejected on two independent grounds:
    #   max_tile_reuse     - cap on total placements of one tile (0 = no cap)
    #   min_reuse_distance - forbid reuse within this many cells, measured
    #                        as Chebyshev distance (0 = no constraint)
    # Both are best-effort: if every candidate is rejected the closest match
    # is used anyway, so a render never fails for want of variety.
    max_tile_reuse: int = 0
    min_reuse_distance: int = 0

    # How many nearest neighbours to consider per cell. Only matters when a
    # repetition constraint is active, and it is the main lever on how well
    # that constraint actually holds: a cell whose whole pool is exhausted
    # falls back to reusing a tile. Enlarging the pool buys variety at the
    # cost of colour accuracy, because later candidates are worse matches.
    #
    # Measured on 400 tiles / 1600 cells with max_tile_reuse=6:
    #     pool= 10  ->  49 distinct, most-used tile 107x, mean distance 167
    #     pool= 50  -> 132 distinct, most-used tile  22x, mean distance 280
    #     pool=200  -> 263 distinct, most-used tile   7x, mean distance 408
    #
    # 25 is a compromise; raise it when a guide has large flat areas that
    # sit far from the colours the tile library actually covers.
    candidate_pool: int = 25

    # Soft repetition control, 0-100. Rather than rejecting a tile outright,
    # every earlier use of it adds a penalty to its colour distance (see
    # MIN/MAX_REUSE_PENALTY), so good-enough tiles win once the best ones
    # have been used many times. 0 disables it.
    variety: int = 0

    # Candidates considered when variety is on. The penalty can only move a
    # cell onto a tile that is in its shortlist, so this needs to be wider
    # than candidate_pool for the penalty to reach beyond the closest few.
    variety_pool: int = 100

    # Visit cells in a shuffled order instead of row by row. Row-major lets
    # the top rows claim the best tiles and leaves the bottom rows with the
    # leftovers whenever variety or a reuse cap is active; a shuffled order
    # spreads that compromise evenly. Seeded, so a render is reproducible.
    randomize_order: bool = True
    seed: int = 0

    # How far each tile's colours are shifted toward its cell's, 0-100.
    # The shift is per section of the 3x3 grid and additive, so a tile keeps
    # its own detail and contrast while taking on the guide's colour. This
    # is what lets a loosely matched tile read correctly from a distance.
    tint_strength: int = 0

    # Memory budget for prepared tile bitmaps held during compositing.
    # Expressed in megabytes rather than as a tile count because a count
    # that is comfortable at 100x100 (3 MB) would be 49 GB at 2000x2000.
    # A constrained render can touch many hundreds of distinct tiles, and
    # a cache smaller than that thrashes, re-preparing tiles it just
    # evicted, so the budget needs to cover a realistic working set.
    tile_cache_mb: int = 256

    # Explicit override for the number of cached bitmaps. Leave as None to
    # derive it from tile_cache_mb and the tile size.
    tile_cache_size: Optional[int] = None

    def __post_init__(self):
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError(
                f"Tile dimensions must be positive: "
                f"{self.tile_width}x{self.tile_height}"
            )
        if self.candidate_pool < 1:
            raise ValueError(
                f"candidate_pool must be at least 1, got {self.candidate_pool}"
            )
        if self.max_tile_reuse < 0:
            raise ValueError("max_tile_reuse cannot be negative")
        if not 0 <= self.variety <= 100:
            raise ValueError(f"variety must be 0-100, got {self.variety}")
        if self.variety_pool < 1:
            raise ValueError(
                f"variety_pool must be at least 1, got {self.variety_pool}"
            )
        if not 0 <= self.tint_strength <= 100:
            raise ValueError(
                f"tint_strength must be 0-100, got {self.tint_strength}"
            )
        if self.min_reuse_distance < 0:
            raise ValueError("min_reuse_distance cannot be negative")
        if self.tile_cache_mb <= 0:
            raise ValueError("tile_cache_mb must be positive")
        if self.tile_cache_size is not None and self.tile_cache_size < 1:
            raise ValueError("tile_cache_size must be at least 1")

    def resolved_tile_cache_size(self) -> int:
        """How many tile bitmaps fit in the configured memory budget."""
        if self.tile_cache_size is not None:
            return self.tile_cache_size
        bytes_per_tile = self.tile_width * self.tile_height * 3
        return max(1, (self.tile_cache_mb * 1024 * 1024) // bytes_per_tile)

    @property
    def reuse_penalty(self) -> float:
        """Colour-distance penalty added per earlier use of a tile.

        0 when variety is off, else logarithmic from MIN_REUSE_PENALTY at
        variety 1 to MAX_REUSE_PENALTY at variety 100.
        """
        if self.variety == 0:
            return 0.0
        ratio = MAX_REUSE_PENALTY / MIN_REUSE_PENALTY
        return MIN_REUSE_PENALTY * ratio ** ((self.variety - 1) / 99)

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (self.tile_width, self.tile_height)

    @property
    def tile_aspect_ratio(self) -> float:
        return self.tile_width / self.tile_height


@dataclass
class TilePlacement:
    """One tile assigned to one grid cell."""

    row: int
    col: int
    tile_data: TileData
    distance: float
    #  True when repetition constraints could not be satisfied and the
    #  closest match was used regardless.
    constraint_relaxed: bool = False
    #  The cell's nine (r, g, b) section colours, row-major, for tinting.
    target_colors: Optional[Sequence[Tuple[int, int, int]]] = None

    @property
    def image_path(self) -> str:
        return self.tile_data.image_path

    def __repr__(self):
        return (f"TilePlacement(row={self.row}, col={self.col}, "
                f"tile='{os.path.basename(self.image_path)}', "
                f"distance={self.distance:.2f})")


@dataclass
class RenderStats:
    """Summary of a completed plan."""

    total_cells: int = 0
    distinct_tiles: int = 0
    relaxed_cells: int = 0
    mean_distance: float = 0.0
    max_tile_uses: int = 0

    def __repr__(self):
        return (f"RenderStats(cells={self.total_cells}, "
                f"distinct={self.distinct_tiles}, "
                f"relaxed={self.relaxed_cells}, "
                f"mean_distance={self.mean_distance:.2f}, "
                f"max_uses={self.max_tile_uses})")


def tint_tile(tile: Image.Image,
              target_colors: Sequence[Tuple[int, int, int]],
              strength: float) -> Image.Image:
    """Shift a tile's colours toward nine target section colours.

    The tile's own 3x3 section averages are measured, the difference to the
    targets is interpolated smoothly across the tile, and a `strength`
    fraction (0-1) of it is added to every pixel. Adding rather than
    blending keeps the tile's detail: at full strength its colours move to
    the targets, but its texture is untouched.
    """
    if strength <= 0:
        return tile
    pixels = np.asarray(tile, dtype=np.float32)
    height, width = pixels.shape[:2]
    current = cv2.resize(pixels, (3, 3), interpolation=cv2.INTER_AREA)
    target = np.asarray(target_colors, dtype=np.float32).reshape(3, 3, 3)
    shift = cv2.resize(target - current, (width, height),
                       interpolation=cv2.INTER_LINEAR)
    tinted = pixels + min(strength, 1.0) * shift
    return Image.fromarray(np.clip(np.rint(tinted), 0, 255).astype(np.uint8))


class _TileBitmapCache:
    """Bounded LRU cache of tile bitmaps at final size.

    Tiles repeat across a mosaic, and preparing one costs a decode plus a
    crop and resize, so caching matters. The cache is bounded because a
    large library at 100x100x3 bytes per tile would otherwise grow without
    limit over a long render.
    """

    def __init__(self, maxsize: int):
        self._maxsize = max(1, maxsize)
        self._entries: "OrderedDict[str, Image.Image]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Image.Image]:
        if key in self._entries:
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        return None

    def put(self, key: str, image: Image.Image) -> None:
        self._entries[key] = image
        self._entries.move_to_end(key)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    def __len__(self):
        return len(self._entries)


class MosaicRenderer:
    """Assembles a mosaic from a guide image and a tile database."""

    def __init__(self, database: TileDatabase,
                 config: Optional[RenderConfig] = None,
                 preprocessor: Optional[TilePreprocessor] = None):
        """
        Args:
            database: A TileDatabase with its index already built
            config: Render settings; defaults to 100x100 tiles, no repetition
                limits
            preprocessor: Used to crop tiles to the tile aspect ratio. Should
                be the same one the database was loaded with so that the
                pixels placed match the colours indexed. If None, tiles are
                centre-cropped to the tile aspect ratio on the fly.
        """
        self._database = database
        self.config = config or RenderConfig()
        self._preprocessor = preprocessor
        self._crop_calculator = CropCalculator(self.config.tile_aspect_ratio)
        self._cache = _TileBitmapCache(self.config.resolved_tile_cache_size())

    # ------------------------------------------------------------------
    # Phase 1: planning
    # ------------------------------------------------------------------

    def plan(self, guide: GuideImage,
             progress_callback: Optional[ProgressCallback] = None,
             should_cancel: Optional[CancelCheck] = None
             ) -> List[TilePlacement]:
        """
        Choose a tile for every cell of the guide grid.

        Cells are visited in a seeded shuffled order when
        config.randomize_order is set, otherwise row by row. Either way the
        result comes back in row-major order.

        Args:
            guide: The analysed guide image
            progress_callback: Called as (completed, total, message)
            should_cancel: Polled periodically; aborts when it returns True

        Returns:
            One TilePlacement per grid cell, in row-major order

        Raises:
            RuntimeError: If the database has no tiles or no index
            RenderCancelled: If should_cancel returns True
        """
        if self._database.size() == 0:
            raise RuntimeError("Cannot render: the tile database is empty")
        if not self._database.is_ready():
            raise RuntimeError(
                "Cannot render: the tile database index has not been built. "
                "Call build_index() first."
            )

        rows, cols = guide.grid_dimensions
        total = rows * cols

        # A reuse cap can be arithmetically impossible: with N tiles capped
        # at M uses, only N*M cells can be filled without exceeding it. Say
        # so rather than quietly overshooting.
        cap = self.config.max_tile_reuse
        if cap and cap * self._database.size() < total:
            logger.warning(
                "max_tile_reuse=%d cannot be honoured: %d tiles x %d uses = "
                "%d placements for %d cells. The cap will be exceeded; add "
                "more tiles or raise the cap.",
                cap, self._database.size(), cap,
                cap * self._database.size(), total
            )

        cells = list(guide.iter_cells())
        if self.config.randomize_order:
            rng = np.random.default_rng(self.config.seed)
            cells = [cells[i] for i in rng.permutation(len(cells))]

        placements: List[Optional[TilePlacement]] = [None] * total
        # grid[row][col] -> image_path, for the adjacency lookup
        grid: List[List[Optional[str]]] = [[None] * cols for _ in range(rows)]
        usage: Dict[str, int] = {}

        for index, cell in enumerate(cells):
            if should_cancel is not None and index % 64 == 0 and should_cancel():
                raise RenderCancelled("Cancelled while matching tiles")

            placement = self._choose_tile(cell, grid, usage)
            placements[cell.row * cols + cell.col] = placement
            grid[cell.row][cell.col] = placement.image_path
            usage[placement.image_path] = usage.get(placement.image_path, 0) + 1

            if progress_callback and (index % 64 == 0 or index + 1 == total):
                progress_callback(index + 1, total, "Matching tiles...")

        return placements

    def _choose_tile(self, cell, grid, usage) -> TilePlacement:
        """Pick the best-scoring candidate that satisfies the hard limits.

        A candidate's score is its colour distance plus the variety penalty
        for each time it has already been used. With variety off, that is
        simply the closest allowed candidate.
        """
        config = self.config
        constrained = config.max_tile_reuse > 0 or config.min_reuse_distance > 0
        penalty = config.reuse_penalty

        # With no limits and no penalty a single neighbour is all we need.
        k = 1
        if constrained:
            k = config.candidate_pool
        if penalty:
            k = max(k, config.variety_pool)
        colors = cell.get_colors()
        candidates = self._database.find_k_nearest_neighbors(colors, k=k)

        if not candidates:
            raise RuntimeError("Tile search returned no candidates")

        targets = [c.as_tuple() for c in colors]

        allowed = [r for r in candidates
                   if not constrained or self._is_allowed(
                       r.tile_data.image_path, cell.row, cell.col, grid, usage)]
        if allowed:
            best = min(allowed, key=lambda r: (
                r.distance + penalty * usage.get(r.tile_data.image_path, 0)))
            return TilePlacement(cell.row, cell.col, best.tile_data,
                                 best.distance, target_colors=targets)

        # Every candidate was rejected by a hard limit (without limits every
        # candidate is allowed). Falling back to the closest match would pile
        # more placements onto the tile that is already the most overused,
        # so prefer the least-used candidate instead and break ties on
        # colour distance. This keeps the reuse cap roughly honoured even
        # when the pool is too small to satisfy it exactly.
        best = min(candidates,
                   key=lambda r: (usage.get(r.tile_data.image_path, 0),
                                  r.distance))
        return TilePlacement(cell.row, cell.col, best.tile_data,
                             best.distance, constraint_relaxed=True,
                             target_colors=targets)

    def _is_allowed(self, path: str, row: int, col: int, grid, usage) -> bool:
        """Check a candidate against the reuse cap and adjacency radius."""
        config = self.config

        if config.max_tile_reuse and usage.get(path, 0) >= config.max_tile_reuse:
            return False

        distance = config.min_reuse_distance
        if distance:
            # Scan the whole square: with a shuffled visiting order, cells
            # below and to the right may already be placed too. Unplaced
            # cells, this one included, hold None and never match.
            row_start = max(0, row - distance)
            row_end = min(len(grid) - 1, row + distance)
            col_start = max(0, col - distance)
            col_end = min(len(grid[0]) - 1, col + distance)
            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    if grid[r][c] == path:
                        return False

        return True

    # ------------------------------------------------------------------
    # Phase 2: compositing
    # ------------------------------------------------------------------

    def composite(self, placements: List[TilePlacement],
                  grid_dimensions: Tuple[int, int],
                  progress_callback: Optional[ProgressCallback] = None,
                  should_cancel: Optional[CancelCheck] = None
                  ) -> Image.Image:
        """
        Paste the planned tiles into a single image.

        Args:
            placements: Output of plan()
            grid_dimensions: (rows, cols) of the guide grid
            progress_callback: Called as (completed, total, message)
            should_cancel: Polled periodically; aborts when it returns True

        Returns:
            The finished mosaic

        Raises:
            ValueError: If the canvas would have zero area
            RenderCancelled: If should_cancel returns True
        """
        rows, cols = grid_dimensions
        tile_w, tile_h = self.config.tile_size
        canvas_size = (cols * tile_w, rows * tile_h)

        if canvas_size[0] <= 0 or canvas_size[1] <= 0:
            raise ValueError(
                f"Cannot composite an empty {rows}x{cols} grid"
            )

        canvas = Image.new('RGB', canvas_size)
        total = len(placements)

        for index, placement in enumerate(placements):
            if should_cancel is not None and index % 64 == 0 and should_cancel():
                raise RenderCancelled("Cancelled while assembling the mosaic")

            bitmap = self._get_tile_bitmap(placement.image_path)
            if bitmap is None:
                continue  # Unreadable tile; leave the cell black.
            if self.config.tint_strength and placement.target_colors:
                # Tinted per cell, after the cache, so the cached bitmap
                # stays the untouched original shared by every placement.
                bitmap = tint_tile(bitmap, placement.target_colors,
                                   self.config.tint_strength / 100)
            canvas.paste(bitmap, (placement.col * tile_w,
                                  placement.row * tile_h))

            if progress_callback and (index % 64 == 0 or index + 1 == total):
                progress_callback(index + 1, total, "Assembling mosaic...")

        return canvas

    def _get_tile_bitmap(self, image_path: str) -> Optional[Image.Image]:
        """Load a tile, crop it to the tile aspect ratio and resize to fit."""
        cached = self._cache.get(image_path)
        if cached is not None:
            return cached

        try:
            prepared = self._prepare_tile(image_path)
        except Exception as e:
            logger.warning(f"Skipping unreadable tile {image_path}: {e}")
            return None

        self._cache.put(image_path, prepared)
        return prepared

    def _prepare_tile(self, image_path: str) -> Image.Image:
        """Produce one tile bitmap at exactly the configured tile size."""
        if self._preprocessor is not None:
            # The preprocessor crops to the target aspect ratio but does not
            # resize, so its output is aspect-correct at an arbitrary size.
            image = self._preprocessor.preprocess_image(image_path)
        else:
            image = open_image(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.crop(
                self._crop_calculator.calculate_crop(image.size)
            )

        if image.size != self.config.tile_size:
            image = image.resize(self.config.tile_size, Image.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    # ------------------------------------------------------------------
    # Both phases
    # ------------------------------------------------------------------

    def render(self, guide: GuideImage,
               progress_callback: Optional[ProgressCallback] = None,
               should_cancel: Optional[CancelCheck] = None
               ) -> Image.Image:
        """Plan and composite in one call."""
        placements = self.plan(guide, progress_callback, should_cancel)
        return self.composite(placements, guide.grid_dimensions,
                              progress_callback, should_cancel)

    @staticmethod
    def summarize(placements: List[TilePlacement]) -> RenderStats:
        """Describe a plan: variety, match quality and constraint pressure."""
        if not placements:
            return RenderStats()

        usage: Dict[str, int] = {}
        for placement in placements:
            usage[placement.image_path] = usage.get(placement.image_path, 0) + 1

        return RenderStats(
            total_cells=len(placements),
            distinct_tiles=len(usage),
            relaxed_cells=sum(1 for p in placements if p.constraint_relaxed),
            mean_distance=sum(p.distance for p in placements) / len(placements),
            max_tile_uses=max(usage.values()),
        )
