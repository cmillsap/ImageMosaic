"""
Guide Image Module

This module provides functionality to load a guide image, divide it into a grid
of cells matching the tile aspect ratio, and analyze each cell's colors using the
same 3x3 section approach as ImageTileAnalyzer.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Iterator
from PIL import Image
import os

from tile_analyzer import BoundingBox, ColorAverage, SectionData, ImageTileAnalyzer


@dataclass
class GridCellData:
    """Represents a single cell in the guide image grid with its 3x3 color analysis."""
    row: int
    col: int
    bounding_box: BoundingBox
    sections: List[SectionData]

    def get_colors(self) -> List[ColorAverage]:
        """Return the 9 average colors, compatible with TileDatabase.find_nearest_neighbor()."""
        return [section.average_color for section in self.sections]

    def __repr__(self):
        return (f"GridCellData(row={self.row}, col={self.col}, "
                f"box={self.bounding_box}, sections={len(self.sections)})")


class GuideImage:
    """
    Loads a guide image, divides it into a grid of cells matching the tile
    dimensions, and analyzes each cell's colors using 3x3 section analysis.

    The guide image is stretched to fit the exact tile grid dimensions so no
    content is lost.
    """

    def __init__(self, image_path: str, output_width_px: int, output_height_px: int,
                 tile_width: int, tile_height: int):
        """
        Initialize GuideImage by loading, resizing, and analyzing the guide image.

        Args:
            image_path: Path to the guide image file
            output_width_px: Desired output width in pixels
            output_height_px: Desired output height in pixels
            tile_width: Width of each tile in pixels
            tile_height: Height of each tile in pixels

        Raises:
            FileNotFoundError: If image_path does not exist
            ValueError: If dimensions are invalid
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Guide image not found: {image_path}")

        if tile_width <= 0 or tile_height <= 0:
            raise ValueError(f"Tile dimensions must be positive: {tile_width}x{tile_height}")

        if output_width_px < tile_width or output_height_px < tile_height:
            raise ValueError(
                f"Output dimensions ({output_width_px}x{output_height_px}) must be at least "
                f"one tile ({tile_width}x{tile_height})"
            )

        self._num_cols = output_width_px // tile_width
        self._num_rows = output_height_px // tile_height
        self._tile_width = tile_width
        self._tile_height = tile_height
        self._used_width = self._num_cols * tile_width
        self._used_height = self._num_rows * tile_height
        self._remainder_x = output_width_px - self._used_width
        self._remainder_y = output_height_px - self._used_height

        resized_img = self._load_and_resize(image_path)
        self._grid = self._analyze_grid(resized_img)

    def _load_and_resize(self, image_path: str) -> Image.Image:
        """Open the guide image, convert to RGB, and resize to the grid area."""
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((self._used_width, self._used_height), Image.LANCZOS)
        return img

    def _analyze_grid(self, img: Image.Image) -> List[List[GridCellData]]:
        """Analyze all cells in the grid, returning a 2D list of GridCellData."""
        analyzer = ImageTileAnalyzer()
        grid = []
        for row in range(self._num_rows):
            row_cells = []
            for col in range(self._num_cols):
                cell = self._analyze_cell(img, row, col, analyzer)
                row_cells.append(cell)
            grid.append(row_cells)
        return grid

    def _analyze_cell(self, img: Image.Image, row: int, col: int,
                      analyzer: ImageTileAnalyzer) -> GridCellData:
        """Crop a single cell from the image and perform 3x3 section analysis."""
        x = col * self._tile_width
        y = row * self._tile_height
        cell_img = img.crop((x, y, x + self._tile_width, y + self._tile_height))

        sections = analyzer._create_sections(cell_img)

        bbox = BoundingBox(x=x, y=y, width=self._tile_width, height=self._tile_height)
        return GridCellData(row=row, col=col, bounding_box=bbox, sections=sections)

    def get_cell(self, row: int, col: int) -> GridCellData:
        """
        Access a cell by grid position with bounds checking.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)

        Returns:
            GridCellData for the specified position

        Raises:
            IndexError: If row or col is out of bounds
        """
        if not (0 <= row < self._num_rows and 0 <= col < self._num_cols):
            raise IndexError(
                f"Cell ({row}, {col}) out of bounds for grid "
                f"{self._num_rows}x{self._num_cols}"
            )
        return self._grid[row][col]

    def get_cell_colors(self, row: int, col: int) -> List[ColorAverage]:
        """Shortcut to get the 9 colors for a cell."""
        return self.get_cell(row, col).get_colors()

    def iter_cells(self) -> Iterator[GridCellData]:
        """Iterate over all cells in row-major order."""
        for row in self._grid:
            for cell in row:
                yield cell

    @property
    def grid_dimensions(self) -> Tuple[int, int]:
        """Return (num_rows, num_cols)."""
        return (self._num_rows, self._num_cols)

    @property
    def remainder_pixels(self) -> Tuple[int, int]:
        """Return (remainder_x, remainder_y) pixels not covered by tiles."""
        return (self._remainder_x, self._remainder_y)

    def __repr__(self):
        return (f"GuideImage(grid={self._num_rows}x{self._num_cols}, "
                f"tile={self._tile_width}x{self._tile_height}, "
                f"used={self._used_width}x{self._used_height})")
