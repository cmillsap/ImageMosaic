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
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from guide_image import GuideImage
from tile_analyzer import TileData
from tile_database import TileDatabase
from tile_preprocessor import CropCalculator, TilePreprocessor

logger = logging.getLogger(__name__)

# Called as callback(completed, total, message).
ProgressCallback = Callable[[int, int, str], None]


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

    # Number of prepared tile bitmaps held in memory during compositing.
    tile_cache_size: int = 512

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
        if self.min_reuse_distance < 0:
            raise ValueError("min_reuse_distance cannot be negative")

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
        self._cache = _TileBitmapCache(self.config.tile_cache_size)

    # ------------------------------------------------------------------
    # Phase 1: planning
    # ------------------------------------------------------------------

    def plan(self, guide: GuideImage,
             progress_callback: Optional[ProgressCallback] = None
             ) -> List[TilePlacement]:
        """
        Choose a tile for every cell of the guide grid.

        Cells are visited in row-major order, which is what makes the
        adjacency constraint meaningful: by the time a cell is considered,
        its neighbours above and to the left are already placed.

        Args:
            guide: The analysed guide image
            progress_callback: Called as (completed, total, message)

        Returns:
            One TilePlacement per grid cell, in row-major order

        Raises:
            RuntimeError: If the database has no tiles or no index
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

        placements: List[TilePlacement] = []
        # grid[row][col] -> image_path, for the adjacency lookup
        grid: List[List[Optional[str]]] = [[None] * cols for _ in range(rows)]
        usage: Dict[str, int] = {}

        for index, cell in enumerate(guide.iter_cells()):
            placement = self._choose_tile(cell, grid, usage)
            placements.append(placement)
            grid[cell.row][cell.col] = placement.image_path
            usage[placement.image_path] = usage.get(placement.image_path, 0) + 1

            if progress_callback and (index % 64 == 0 or index + 1 == total):
                progress_callback(index + 1, total, "Matching tiles...")

        return placements

    def _choose_tile(self, cell, grid, usage) -> TilePlacement:
        """Pick the closest candidate that satisfies the repetition rules."""
        config = self.config
        constrained = config.max_tile_reuse > 0 or config.min_reuse_distance > 0

        # Without constraints a single neighbour is all that is needed.
        k = config.candidate_pool if constrained else 1
        candidates = self._database.find_k_nearest_neighbors(
            cell.get_colors(), k=k
        )

        if not candidates:
            raise RuntimeError("Tile search returned no candidates")

        if constrained:
            for result in candidates:
                path = result.tile_data.image_path
                if self._is_allowed(path, cell.row, cell.col, grid, usage):
                    return TilePlacement(cell.row, cell.col,
                                         result.tile_data, result.distance)

            # Every candidate was rejected. Falling back to the closest match
            # would pile more placements onto the tile that is already the
            # most overused, so prefer the least-used candidate instead and
            # break ties on colour distance. This keeps the reuse cap roughly
            # honoured even when the pool is too small to satisfy it exactly.
            best = min(candidates,
                       key=lambda r: (usage.get(r.tile_data.image_path, 0),
                                      r.distance))
            return TilePlacement(cell.row, cell.col, best.tile_data,
                                 best.distance, constraint_relaxed=True)

        # Unconstrained: the nearest neighbour is the answer.
        best = candidates[0]
        return TilePlacement(cell.row, cell.col, best.tile_data, best.distance)

    def _is_allowed(self, path: str, row: int, col: int, grid, usage) -> bool:
        """Check a candidate against the reuse cap and adjacency radius."""
        config = self.config

        if config.max_tile_reuse and usage.get(path, 0) >= config.max_tile_reuse:
            return False

        distance = config.min_reuse_distance
        if distance:
            # Only cells already placed can conflict, so scanning the full
            # square is wasteful but simple, and the radius is small.
            row_start = max(0, row - distance)
            col_start = max(0, col - distance)
            col_end = min(len(grid[0]) - 1, col + distance)
            for r in range(row_start, row + 1):
                for c in range(col_start, col_end + 1):
                    if r == row and c >= col:
                        break
                    if grid[r][c] == path:
                        return False

        return True

    # ------------------------------------------------------------------
    # Phase 2: compositing
    # ------------------------------------------------------------------

    def composite(self, placements: List[TilePlacement],
                  grid_dimensions: Tuple[int, int],
                  progress_callback: Optional[ProgressCallback] = None
                  ) -> Image.Image:
        """
        Paste the planned tiles into a single image.

        Args:
            placements: Output of plan()
            grid_dimensions: (rows, cols) of the guide grid
            progress_callback: Called as (completed, total, message)

        Returns:
            The finished mosaic

        Raises:
            ValueError: If the canvas would have zero area
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
            bitmap = self._get_tile_bitmap(placement.image_path)
            if bitmap is None:
                continue  # Unreadable tile; leave the cell black.
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
            image = Image.open(image_path)
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
               progress_callback: Optional[ProgressCallback] = None
               ) -> Image.Image:
        """Plan and composite in one call."""
        placements = self.plan(guide, progress_callback)
        return self.composite(placements, guide.grid_dimensions,
                              progress_callback)

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
