"""
Test suite for the guide_image module.

Run with: pytest test_guide_image.py -v
"""

import pytest
import os
import tempfile
from PIL import Image
from tile_analyzer import BoundingBox, ColorAverage, SectionData
from guide_image import GridCellData, GuideImage


class TestGridCellData:
    """Tests for GridCellData dataclass."""

    def _make_sections(self, color=(128, 128, 128)):
        """Helper to create 9 sections with a given color."""
        sections = []
        for i in range(9):
            bbox = BoundingBox(x=i * 10, y=i * 10, width=10, height=10)
            avg = ColorAverage(r=color[0], g=color[1], b=color[2])
            sections.append(SectionData(bounding_box=bbox, average_color=avg))
        return sections

    def test_creation(self):
        sections = self._make_sections()
        bbox = BoundingBox(x=0, y=0, width=30, height=30)
        cell = GridCellData(row=1, col=2, bounding_box=bbox, sections=sections)

        assert cell.row == 1
        assert cell.col == 2
        assert cell.bounding_box == bbox
        assert len(cell.sections) == 9

    def test_get_colors_returns_9_color_averages(self):
        sections = self._make_sections(color=(10, 20, 30))
        bbox = BoundingBox(x=0, y=0, width=30, height=30)
        cell = GridCellData(row=0, col=0, bounding_box=bbox, sections=sections)

        colors = cell.get_colors()
        assert len(colors) == 9
        assert all(isinstance(c, ColorAverage) for c in colors)
        assert all(c.r == 10 and c.g == 20 and c.b == 30 for c in colors)

    def test_repr(self):
        sections = self._make_sections()
        bbox = BoundingBox(x=0, y=0, width=30, height=30)
        cell = GridCellData(row=0, col=0, bounding_box=bbox, sections=sections)
        r = repr(cell)
        assert "GridCellData" in r
        assert "row=0" in r
        assert "col=0" in r


class TestGuideImageInit:
    """Tests for GuideImage constructor and dimension computation."""

    @pytest.fixture
    def temp_image_path(self):
        """Create a temporary test image file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()
        yield temp_file.name
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    def _save_solid(self, path, width, height, color=(100, 100, 100)):
        img = Image.new('RGB', (width, height), color)
        img.save(path)

    def test_grid_dimension_computation(self, temp_image_path):
        self._save_solid(temp_image_path, 200, 200)
        guide = GuideImage(temp_image_path, output_width_px=320, output_height_px=240,
                           tile_width=32, tile_height=24)
        assert guide.grid_dimensions == (10, 10)

    def test_grid_dimension_non_divisible(self, temp_image_path):
        self._save_solid(temp_image_path, 200, 200)
        guide = GuideImage(temp_image_path, output_width_px=100, output_height_px=100,
                           tile_width=30, tile_height=30)
        # 100 // 30 = 3
        assert guide.grid_dimensions == (3, 3)

    def test_remainder_pixels(self, temp_image_path):
        self._save_solid(temp_image_path, 200, 200)
        guide = GuideImage(temp_image_path, output_width_px=100, output_height_px=100,
                           tile_width=30, tile_height=30)
        # 100 - 3*30 = 10
        assert guide.remainder_pixels == (10, 10)

    def test_remainder_zero_when_exact(self, temp_image_path):
        self._save_solid(temp_image_path, 200, 200)
        guide = GuideImage(temp_image_path, output_width_px=90, output_height_px=90,
                           tile_width=30, tile_height=30)
        assert guide.remainder_pixels == (0, 0)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Guide image not found"):
            GuideImage("nonexistent.png", 100, 100, 10, 10)

    def test_invalid_tile_dimensions(self, temp_image_path):
        self._save_solid(temp_image_path, 200, 200)
        with pytest.raises(ValueError, match="Tile dimensions must be positive"):
            GuideImage(temp_image_path, 100, 100, 0, 10)
        with pytest.raises(ValueError, match="Tile dimensions must be positive"):
            GuideImage(temp_image_path, 100, 100, 10, -5)

    def test_output_too_small_for_one_tile(self, temp_image_path):
        self._save_solid(temp_image_path, 200, 200)
        with pytest.raises(ValueError, match="Output dimensions"):
            GuideImage(temp_image_path, 5, 100, 10, 10)

    def test_repr(self, temp_image_path):
        self._save_solid(temp_image_path, 200, 200)
        guide = GuideImage(temp_image_path, 90, 90, 30, 30)
        r = repr(guide)
        assert "GuideImage" in r
        assert "3x3" in r


class TestGuideImageAnalysis:
    """Tests for guide image analysis with known colors."""

    @pytest.fixture
    def temp_image_path(self):
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()
        yield temp_file.name
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    def test_solid_color_all_cells_match(self, temp_image_path):
        """A solid red guide image should produce cells with all-red sections."""
        img = Image.new('RGB', (90, 90), (255, 0, 0))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 90, 90, 30, 30)
        assert guide.grid_dimensions == (3, 3)

        for cell in guide.iter_cells():
            colors = cell.get_colors()
            assert len(colors) == 9
            for c in colors:
                assert c.r == 255
                assert c.g == 0
                assert c.b == 0

    def test_four_quadrants(self, temp_image_path):
        """2x2 grid where each cell is a different solid color."""
        img = Image.new('RGB', (60, 60))
        pixels = img.load()

        # Top-left (0,0): red
        # Top-right (0,1): green
        # Bottom-left (1,0): blue
        # Bottom-right (1,1): white
        quadrant_colors = {
            (0, 0): (255, 0, 0),
            (0, 1): (0, 255, 0),
            (1, 0): (0, 0, 255),
            (1, 1): (255, 255, 255),
        }
        for y in range(60):
            for x in range(60):
                qr = y // 30
                qc = x // 30
                pixels[x, y] = quadrant_colors[(qr, qc)]

        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 60, 60, 30, 30)
        assert guide.grid_dimensions == (2, 2)

        for (row, col), expected in quadrant_colors.items():
            colors = guide.get_cell_colors(row, col)
            assert len(colors) == 9
            for c in colors:
                assert c.as_tuple() == expected, (
                    f"Cell ({row},{col}): expected {expected}, got {c.as_tuple()}"
                )

    def test_get_cell(self, temp_image_path):
        img = Image.new('RGB', (60, 60), (50, 100, 150))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 60, 60, 30, 30)
        cell = guide.get_cell(0, 1)
        assert cell.row == 0
        assert cell.col == 1
        assert cell.bounding_box.x == 30
        assert cell.bounding_box.y == 0

    def test_get_cell_out_of_bounds(self, temp_image_path):
        img = Image.new('RGB', (60, 60), (50, 100, 150))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 60, 60, 30, 30)
        with pytest.raises(IndexError, match="out of bounds"):
            guide.get_cell(2, 0)
        with pytest.raises(IndexError, match="out of bounds"):
            guide.get_cell(0, 2)
        with pytest.raises(IndexError, match="out of bounds"):
            guide.get_cell(-1, 0)

    def test_iter_cells_count(self, temp_image_path):
        img = Image.new('RGB', (90, 60), (0, 0, 0))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 90, 60, 30, 30)
        cells = list(guide.iter_cells())
        assert len(cells) == 6  # 2 rows x 3 cols

    def test_iter_cells_row_major_order(self, temp_image_path):
        img = Image.new('RGB', (90, 60), (0, 0, 0))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 90, 60, 30, 30)
        cells = list(guide.iter_cells())
        expected_positions = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
        ]
        for cell, (er, ec) in zip(cells, expected_positions):
            assert (cell.row, cell.col) == (er, ec)

    def test_cell_bounding_boxes(self, temp_image_path):
        img = Image.new('RGB', (60, 40), (0, 0, 0))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 60, 40, 20, 20)
        # 3 cols x 2 rows
        cell_00 = guide.get_cell(0, 0)
        assert cell_00.bounding_box.x == 0
        assert cell_00.bounding_box.y == 0
        assert cell_00.bounding_box.width == 20
        assert cell_00.bounding_box.height == 20

        cell_01 = guide.get_cell(0, 1)
        assert cell_01.bounding_box.x == 20

        cell_10 = guide.get_cell(1, 0)
        assert cell_10.bounding_box.y == 20

    def test_sections_have_9_entries(self, temp_image_path):
        img = Image.new('RGB', (90, 90), (128, 64, 32))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 90, 90, 30, 30)
        for cell in guide.iter_cells():
            assert len(cell.sections) == 9
            for section in cell.sections:
                assert isinstance(section, SectionData)


class TestGuideImageEdgeCases:
    """Edge case tests: RGBA/grayscale conversion, non-divisible dims, aspect ratio."""

    @pytest.fixture
    def temp_image_path(self):
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()
        yield temp_file.name
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    def test_rgba_image_conversion(self, temp_image_path):
        img = Image.new('RGBA', (90, 90), (200, 100, 50, 128))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 90, 90, 30, 30)
        colors = guide.get_cell_colors(0, 0)
        assert len(colors) == 9
        # After RGBA->RGB conversion, alpha is composited on black by PIL
        # The resulting color depends on PIL's conversion behavior
        for c in colors:
            assert isinstance(c, ColorAverage)

    def test_grayscale_image_conversion(self, temp_image_path):
        img = Image.new('L', (90, 90), 128)
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 90, 90, 30, 30)
        colors = guide.get_cell_colors(0, 0)
        assert len(colors) == 9
        # Grayscale -> RGB means R=G=B
        for c in colors:
            assert c.r == c.g == c.b

    def test_non_divisible_output_dimensions(self, temp_image_path):
        """Output dims not evenly divisible by tile size should use floor division."""
        img = Image.new('RGB', (200, 200), (50, 50, 50))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 100, 100, 33, 33)
        # 100 // 33 = 3
        assert guide.grid_dimensions == (3, 3)
        assert guide.remainder_pixels == (1, 1)

    def test_aspect_ratio_mismatch(self, temp_image_path):
        """Guide image with different aspect ratio than output should stretch."""
        # Tall narrow source image
        img = Image.new('RGB', (50, 200), (255, 0, 0))
        img.save(temp_image_path)

        # Wide output
        guide = GuideImage(temp_image_path, 300, 100, 30, 25)
        assert guide.grid_dimensions == (4, 10)

        # Should still analyze without error, all cells should have red-ish colors
        for cell in guide.iter_cells():
            colors = cell.get_colors()
            assert len(colors) == 9

    def test_single_tile_grid(self, temp_image_path):
        """Minimum grid: 1x1."""
        img = Image.new('RGB', (100, 100), (10, 20, 30))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 50, 50, 50, 50)
        assert guide.grid_dimensions == (1, 1)
        cell = guide.get_cell(0, 0)
        colors = cell.get_colors()
        assert len(colors) == 9
        for c in colors:
            assert c.r == 10
            assert c.g == 20
            assert c.b == 30

    def test_rectangular_tiles(self, temp_image_path):
        """Tiles wider than tall."""
        img = Image.new('RGB', (200, 200), (0, 128, 255))
        img.save(temp_image_path)

        guide = GuideImage(temp_image_path, 120, 60, 40, 20)
        assert guide.grid_dimensions == (3, 3)
        for cell in guide.iter_cells():
            assert cell.bounding_box.width == 40
            assert cell.bounding_box.height == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
