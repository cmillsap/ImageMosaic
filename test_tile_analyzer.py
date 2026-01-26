"""
Test suite for the tile_analyzer module.

Run with: pytest test_tile_analyzer.py -v
"""

import pytest
import os
import tempfile
from PIL import Image
from tile_analyzer import (
    BoundingBox,
    ColorAverage,
    SectionData,
    TileData,
    ImageTileAnalyzer
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_bounding_box_creation(self):
        bbox = BoundingBox(x=10, y=20, width=100, height=200)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 200

    def test_bounding_box_repr(self):
        bbox = BoundingBox(x=0, y=0, width=50, height=50)
        assert "BoundingBox" in repr(bbox)
        assert "x=0" in repr(bbox)


class TestColorAverage:
    """Tests for ColorAverage dataclass."""

    def test_color_average_creation(self):
        color = ColorAverage(r=255, g=128, b=64)
        assert color.r == 255
        assert color.g == 128
        assert color.b == 64

    def test_color_as_tuple(self):
        color = ColorAverage(r=100, g=150, b=200)
        assert color.as_tuple() == (100, 150, 200)

    def test_color_repr(self):
        color = ColorAverage(r=0, g=0, b=0)
        assert "ColorAverage" in repr(color)


class TestSectionData:
    """Tests for SectionData dataclass."""

    def test_section_data_creation(self):
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        color = ColorAverage(r=128, g=128, b=128)
        section = SectionData(bounding_box=bbox, average_color=color)

        assert section.bounding_box == bbox
        assert section.average_color == color


class TestTileData:
    """Tests for TileData class."""

    def create_dummy_sections(self):
        """Helper to create 9 dummy sections."""
        sections = []
        for i in range(9):
            bbox = BoundingBox(x=i*10, y=i*10, width=10, height=10)
            color = ColorAverage(r=i*10, g=i*10, b=i*10)
            sections.append(SectionData(bbox, color))
        return sections

    def test_tile_data_creation(self):
        sections = self.create_dummy_sections()
        tile_data = TileData("test.jpg", sections)

        assert tile_data.image_path == "test.jpg"
        assert len(tile_data.sections) == 9

    def test_tile_data_requires_9_sections(self):
        with pytest.raises(ValueError, match="Expected 9 sections"):
            TileData("test.jpg", [])

        with pytest.raises(ValueError, match="Expected 9 sections"):
            sections = self.create_dummy_sections()[:5]
            TileData("test.jpg", sections)

    def test_get_section_by_grid_position(self):
        sections = self.create_dummy_sections()
        tile_data = TileData("test.jpg", sections)

        # Test corner positions
        assert tile_data.get_section(0, 0) == sections[0]  # Top-left
        assert tile_data.get_section(0, 2) == sections[2]  # Top-right
        assert tile_data.get_section(2, 0) == sections[6]  # Bottom-left
        assert tile_data.get_section(2, 2) == sections[8]  # Bottom-right

        # Test center
        assert tile_data.get_section(1, 1) == sections[4]

    def test_get_section_invalid_position(self):
        sections = self.create_dummy_sections()
        tile_data = TileData("test.jpg", sections)

        with pytest.raises(ValueError, match="Invalid grid position"):
            tile_data.get_section(3, 0)

        with pytest.raises(ValueError, match="Invalid grid position"):
            tile_data.get_section(0, 3)

        with pytest.raises(ValueError, match="Invalid grid position"):
            tile_data.get_section(-1, 0)

    def test_get_all_colors(self):
        sections = self.create_dummy_sections()
        tile_data = TileData("test.jpg", sections)

        colors = tile_data.get_all_colors()
        assert len(colors) == 9
        assert all(isinstance(c, ColorAverage) for c in colors)

    def test_get_all_bounding_boxes(self):
        sections = self.create_dummy_sections()
        tile_data = TileData("test.jpg", sections)

        boxes = tile_data.get_all_bounding_boxes()
        assert len(boxes) == 9
        assert all(isinstance(b, BoundingBox) for b in boxes)


class TestImageTileAnalyzer:
    """Tests for ImageTileAnalyzer class."""

    @pytest.fixture
    def temp_image_path(self):
        """Create a temporary test image."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()
        yield temp_file.name
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    def create_solid_color_image(self, path, width, height, color):
        """Create a solid color test image."""
        img = Image.new('RGB', (width, height), color)
        img.save(path)

    def create_gradient_image(self, path, width, height):
        """Create a gradient test image."""
        img = Image.new('RGB', (width, height))
        pixels = img.load()

        for y in range(height):
            for x in range(width):
                r = int((x / width) * 255)
                g = int((y / height) * 255)
                b = 128
                pixels[x, y] = (r, g, b)

        img.save(path)

    def test_analyze_nonexistent_file(self):
        analyzer = ImageTileAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.analyze_image("nonexistent_file.png")

    def test_analyze_solid_color_image(self, temp_image_path):
        analyzer = ImageTileAnalyzer()
        color = (255, 0, 0)  # Red
        self.create_solid_color_image(temp_image_path, 300, 300, color)

        tile_data = analyzer.analyze_image(temp_image_path)

        assert tile_data.image_path == temp_image_path
        assert len(tile_data.sections) == 9

        # All sections should have the same color
        for section in tile_data.sections:
            assert section.average_color.r == 255
            assert section.average_color.g == 0
            assert section.average_color.b == 0

    def test_analyze_image_sections_equal_size(self, temp_image_path):
        """Test that sections are as equal as possible."""
        analyzer = ImageTileAnalyzer()
        self.create_solid_color_image(temp_image_path, 300, 300, (128, 128, 128))

        tile_data = analyzer.analyze_image(temp_image_path)

        # Each section should be 100x100 for a 300x300 image
        for section in tile_data.sections:
            assert section.bounding_box.width == 100
            assert section.bounding_box.height == 100

    def test_analyze_image_non_divisible_dimensions(self, temp_image_path):
        """Test with dimensions not evenly divisible by 3."""
        analyzer = ImageTileAnalyzer()
        self.create_solid_color_image(temp_image_path, 100, 100, (64, 128, 192))

        tile_data = analyzer.analyze_image(temp_image_path)

        # Check that sections cover the entire image
        boxes = tile_data.get_all_bounding_boxes()

        # Check first row
        assert boxes[0].x == 0 and boxes[0].y == 0
        assert boxes[1].x == 33 and boxes[1].y == 0
        assert boxes[2].x == 66 and boxes[2].y == 0

        # Last section should extend to image edge
        assert boxes[2].x + boxes[2].width == 100
        assert boxes[8].y + boxes[8].height == 100

    def test_analyze_image_different_colors_per_section(self, temp_image_path):
        """Test that different colored sections are detected correctly."""
        analyzer = ImageTileAnalyzer()

        # Create a 90x90 image (divisible by 3)
        width, height = 90, 90
        img = Image.new('RGB', (width, height))
        pixels = img.load()

        # Fill each 30x30 section with different colors
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),     # Row 0
            (255, 255, 0), (255, 0, 255), (0, 255, 255),  # Row 1
            (128, 0, 0), (0, 128, 0), (0, 0, 128)      # Row 2
        ]

        section_idx = 0
        for row in range(3):
            for col in range(3):
                color = colors[section_idx]
                for y in range(row * 30, (row + 1) * 30):
                    for x in range(col * 30, (col + 1) * 30):
                        pixels[x, y] = color
                section_idx += 1

        img.save(temp_image_path)

        tile_data = analyzer.analyze_image(temp_image_path)

        # Check that each section has approximately the expected color
        for i, expected_color in enumerate(colors):
            actual_color = tile_data.sections[i].average_color
            assert actual_color.r == expected_color[0]
            assert actual_color.g == expected_color[1]
            assert actual_color.b == expected_color[2]

    def test_analyze_rgba_image(self, temp_image_path):
        """Test that RGBA images are handled correctly."""
        analyzer = ImageTileAnalyzer()

        # Create RGBA image
        img = Image.new('RGBA', (90, 90), (255, 128, 64, 255))
        img.save(temp_image_path)

        # Should convert to RGB and analyze without error
        tile_data = analyzer.analyze_image(temp_image_path)
        assert len(tile_data.sections) == 9
        assert tile_data.sections[0].average_color.r == 255
        assert tile_data.sections[0].average_color.g == 128
        assert tile_data.sections[0].average_color.b == 64

    def test_analyze_grayscale_image(self, temp_image_path):
        """Test that grayscale images are handled correctly."""
        analyzer = ImageTileAnalyzer()

        # Create grayscale image
        img = Image.new('L', (90, 90), 128)
        img.save(temp_image_path)

        # Should convert to RGB and analyze without error
        tile_data = analyzer.analyze_image(temp_image_path)
        assert len(tile_data.sections) == 9

        # Grayscale should result in R=G=B
        for section in tile_data.sections:
            assert section.average_color.r == section.average_color.g == section.average_color.b

    def test_analyze_rectangular_image(self, temp_image_path):
        """Test with non-square image."""
        analyzer = ImageTileAnalyzer()
        self.create_solid_color_image(temp_image_path, 150, 90, (100, 100, 100))

        tile_data = analyzer.analyze_image(temp_image_path)

        # Should still create 9 sections
        assert len(tile_data.sections) == 9

        # Check that sections have correct dimensions
        # Width sections: 50, 50, 50
        # Height sections: 30, 30, 30
        assert tile_data.sections[0].bounding_box.width == 50
        assert tile_data.sections[0].bounding_box.height == 30

    def test_bounding_boxes_cover_entire_image(self, temp_image_path):
        """Test that all bounding boxes together cover the entire image."""
        analyzer = ImageTileAnalyzer()
        width, height = 100, 100
        self.create_solid_color_image(temp_image_path, width, height, (0, 0, 0))

        tile_data = analyzer.analyze_image(temp_image_path)

        # Check that sections don't overlap and cover entire image
        for row in range(3):
            for col in range(3):
                section = tile_data.get_section(row, col)
                bbox = section.bounding_box

                # Verify bounding box is within image bounds
                assert bbox.x >= 0
                assert bbox.y >= 0
                assert bbox.x + bbox.width <= width
                assert bbox.y + bbox.height <= height

    def test_average_color_calculation(self, temp_image_path):
        """Test that average color is calculated correctly."""
        analyzer = ImageTileAnalyzer()

        # Create a 6x6 image where each section will be 2x2 pixels
        # This allows us to test averaging within a section
        img = Image.new('RGB', (6, 6))
        pixels = img.load()

        # Fill the center section (2x2 pixels at coords 2-3, 2-3) with:
        # 2 black pixels and 2 white pixels
        # Average should be (127, 127, 127)
        for y in range(6):
            for x in range(6):
                pixels[x, y] = (0, 0, 0)  # Default to black

        # Center section pixels (section 4):
        # Top-left and bottom-right = white
        # Top-right and bottom-left = black
        pixels[2, 2] = (255, 255, 255)  # Top-left
        pixels[3, 3] = (255, 255, 255)  # Bottom-right
        pixels[3, 2] = (0, 0, 0)        # Top-right
        pixels[2, 3] = (0, 0, 0)        # Bottom-left

        img.save(temp_image_path)

        tile_data = analyzer.analyze_image(temp_image_path)

        # Center section (index 4) should have average of (127, 127, 127)
        # (2*255 + 2*0) / 4 = 127.5 ≈ 128
        avg_color = tile_data.sections[4].average_color
        assert 127 <= avg_color.r <= 128
        assert 127 <= avg_color.g <= 128
        assert 127 <= avg_color.b <= 128


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow(self, tmp_path):
        """Test analyzing multiple images in a workflow."""
        analyzer = ImageTileAnalyzer()

        # Create multiple test images
        image_paths = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for i, color in enumerate(colors):
            path = tmp_path / f"test_{i}.png"
            img = Image.new('RGB', (90, 90), color)
            img.save(path)
            image_paths.append(str(path))

        # Analyze all images
        tile_data_list = []
        for path in image_paths:
            tile_data = analyzer.analyze_image(path)
            tile_data_list.append(tile_data)

        # Verify results
        assert len(tile_data_list) == 3

        for i, tile_data in enumerate(tile_data_list):
            assert len(tile_data.sections) == 9
            expected_color = colors[i]

            # All sections should match the solid color
            for section in tile_data.sections:
                assert section.average_color.r == expected_color[0]
                assert section.average_color.g == expected_color[1]
                assert section.average_color.b == expected_color[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
