"""
Test suite for the tile_database module.

Run with: pytest test_tile_database.py -v
"""

import pytest
import os
import tempfile
import numpy as np
from PIL import Image
from tile_database import TileDatabase, SearchResult
from tile_analyzer import TileData, ColorAverage, BoundingBox, SectionData, ImageTileAnalyzer


class TestTileDatabase:
    """Tests for TileDatabase class."""

    def create_solid_color_tile(self, path, color):
        """Helper to create a solid color test image."""
        img = Image.new('RGB', (90, 90), color)
        img.save(path)

    def create_tile_data(self, path, color):
        """Helper to create TileData with solid color."""
        sections = []
        for i in range(9):
            bbox = BoundingBox(x=i*30, y=0, width=30, height=30)
            avg_color = ColorAverage(r=color[0], g=color[1], b=color[2])
            sections.append(SectionData(bbox, avg_color))
        return TileData(path, sections)

    def test_database_initialization(self):
        """Test database is initialized empty."""
        db = TileDatabase()
        assert len(db) == 0
        assert db.size() == 0
        assert not db.is_ready()

    def test_add_tile(self):
        """Test adding a tile to the database."""
        db = TileDatabase()
        tile_data = self.create_tile_data("test.jpg", (255, 0, 0))

        db.add_tile(tile_data)
        assert len(db) == 1
        assert not db.is_ready()  # Index not built yet

    def test_add_tile_from_path(self, tmp_path):
        """Test adding a tile from file path."""
        db = TileDatabase()
        img_path = tmp_path / "test.png"
        self.create_solid_color_tile(img_path, (128, 128, 128))

        db.add_tile_from_path(str(img_path))
        assert len(db) == 1

    def test_build_index_empty_database(self):
        """Test building index on empty database raises error."""
        db = TileDatabase()
        with pytest.raises(ValueError, match="no tiles in database"):
            db.build_index()

    def test_build_index(self, tmp_path):
        """Test building the search index."""
        db = TileDatabase()

        # Add some tiles
        for i in range(3):
            img_path = tmp_path / f"test_{i}.png"
            self.create_solid_color_tile(img_path, (i * 50, i * 50, i * 50))
            db.add_tile_from_path(str(img_path))

        assert not db.is_ready()
        db.build_index()
        assert db.is_ready()

    def test_search_without_index_raises_error(self, tmp_path):
        """Test searching without building index raises error."""
        db = TileDatabase()
        img_path = tmp_path / "test.png"
        self.create_solid_color_tile(img_path, (128, 128, 128))
        db.add_tile_from_path(str(img_path))

        query = [ColorAverage(r=128, g=128, b=128) for _ in range(9)]

        with pytest.raises(RuntimeError, match="Index not built"):
            db.find_nearest_neighbor(query)

    def test_find_nearest_neighbor_exact_match(self, tmp_path):
        """Test finding exact color match."""
        db = TileDatabase()

        # Create tiles with different colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, color in enumerate(colors):
            img_path = tmp_path / f"test_{i}.png"
            self.create_solid_color_tile(img_path, color)
            db.add_tile_from_path(str(img_path))

        db.build_index()

        # Query for blue - should match tile 2
        query = [ColorAverage(r=0, g=0, b=255) for _ in range(9)]
        result = db.find_nearest_neighbor(query)

        assert result is not None
        # Check that result has blue color
        colors = result.get_all_colors()
        assert all(c.r == 0 and c.g == 0 and c.b == 255 for c in colors)

    def test_find_k_nearest_neighbors(self, tmp_path):
        """Test finding k nearest neighbors."""
        db = TileDatabase()

        # Create tiles with different shades of red
        colors = [(255, 0, 0), (200, 0, 0), (150, 0, 0), (100, 0, 0)]
        for i, color in enumerate(colors):
            img_path = tmp_path / f"test_{i}.png"
            self.create_solid_color_tile(img_path, color)
            db.add_tile_from_path(str(img_path))

        db.build_index()

        # Query for bright red
        query = [ColorAverage(r=255, g=0, b=0) for _ in range(9)]
        results = db.find_k_nearest_neighbors(query, k=3)

        assert len(results) == 3
        # Results should be sorted by distance
        assert results[0].distance <= results[1].distance <= results[2].distance
        # First result should be the exact match
        assert results[0].distance < 1.0  # Very close to 0

    def test_find_k_nearest_neighbors_k_larger_than_database(self, tmp_path):
        """Test k larger than number of tiles."""
        db = TileDatabase()

        # Add only 2 tiles
        for i in range(2):
            img_path = tmp_path / f"test_{i}.png"
            self.create_solid_color_tile(img_path, (i * 100, 0, 0))
            db.add_tile_from_path(str(img_path))

        db.build_index()

        query = [ColorAverage(r=50, g=0, b=0) for _ in range(9)]
        results = db.find_k_nearest_neighbors(query, k=10)

        # Should only return 2 results
        assert len(results) == 2

    def test_find_neighbors_invalid_query_length(self, tmp_path):
        """Test searching with invalid query color count."""
        db = TileDatabase()
        img_path = tmp_path / "test.png"
        self.create_solid_color_tile(img_path, (128, 128, 128))
        db.add_tile_from_path(str(img_path))
        db.build_index()

        # Query with wrong number of colors
        query = [ColorAverage(r=128, g=128, b=128) for _ in range(5)]

        with pytest.raises(ValueError, match="Expected 9 colors"):
            db.find_nearest_neighbor(query)

    def test_find_neighbors_invalid_k(self, tmp_path):
        """Test searching with invalid k value."""
        db = TileDatabase()
        img_path = tmp_path / "test.png"
        self.create_solid_color_tile(img_path, (128, 128, 128))
        db.add_tile_from_path(str(img_path))
        db.build_index()

        query = [ColorAverage(r=128, g=128, b=128) for _ in range(9)]

        with pytest.raises(ValueError, match="k must be at least 1"):
            db.find_k_nearest_neighbors(query, k=0)

    def test_clear_database(self, tmp_path):
        """Test clearing the database."""
        db = TileDatabase()

        for i in range(3):
            img_path = tmp_path / f"test_{i}.png"
            self.create_solid_color_tile(img_path, (i * 50, 0, 0))
            db.add_tile_from_path(str(img_path))

        db.build_index()
        assert len(db) == 3
        assert db.is_ready()

        db.clear()
        assert len(db) == 0
        assert not db.is_ready()

    def test_get_all_tiles(self, tmp_path):
        """Test getting all tiles."""
        db = TileDatabase()

        for i in range(3):
            img_path = tmp_path / f"test_{i}.png"
            self.create_solid_color_tile(img_path, (i * 50, 0, 0))
            db.add_tile_from_path(str(img_path))

        all_tiles = db.get_all_tiles()
        assert len(all_tiles) == 3
        assert all(isinstance(t, TileData) for t in all_tiles)

    def test_get_tile_by_index(self, tmp_path):
        """Test getting tile by index."""
        db = TileDatabase()

        img_path = tmp_path / "test.png"
        self.create_solid_color_tile(img_path, (255, 0, 0))
        db.add_tile_from_path(str(img_path))

        tile = db.get_tile_by_index(0)
        assert isinstance(tile, TileData)

    def test_get_tile_by_invalid_index(self, tmp_path):
        """Test getting tile with invalid index."""
        db = TileDatabase()

        img_path = tmp_path / "test.png"
        self.create_solid_color_tile(img_path, (255, 0, 0))
        db.add_tile_from_path(str(img_path))

        with pytest.raises(IndexError):
            db.get_tile_by_index(10)

    def test_database_repr(self, tmp_path):
        """Test string representation of database."""
        db = TileDatabase()
        assert "not indexed" in repr(db)

        img_path = tmp_path / "test.png"
        self.create_solid_color_tile(img_path, (255, 0, 0))
        db.add_tile_from_path(str(img_path))
        db.build_index()

        assert "ready" in repr(db)
        assert "tiles=1" in repr(db)


class TestTileDatabaseFolderLoading:
    """Tests for loading tiles from folders."""

    def create_test_folder(self, tmp_path, num_images=5):
        """Helper to create a folder with test images."""
        folder = tmp_path / "tiles"
        folder.mkdir()

        for i in range(num_images):
            img_path = folder / f"tile_{i}.png"
            color = (i * 40, i * 30, i * 20)
            img = Image.new('RGB', (90, 90), color)
            img.save(img_path)

        return folder

    def test_load_tiles_from_folder(self, tmp_path):
        """Test loading tiles from a folder."""
        folder = self.create_test_folder(tmp_path, 5)

        db = TileDatabase()
        count = db.load_tiles_from_folder(str(folder))

        assert count == 5
        assert len(db) == 5
        assert db.is_ready()  # auto_build_index is True by default

    def test_load_tiles_without_auto_build(self, tmp_path):
        """Test loading without auto building index."""
        folder = self.create_test_folder(tmp_path, 3)

        db = TileDatabase()
        count = db.load_tiles_from_folder(str(folder), auto_build_index=False)

        assert count == 3
        assert len(db) == 3
        assert not db.is_ready()

    def test_load_tiles_with_extension_filter(self, tmp_path):
        """Test loading with file extension filter."""
        folder = tmp_path / "tiles"
        folder.mkdir()

        # Create PNG and JPG files
        for i in range(3):
            img = Image.new('RGB', (90, 90), (i * 50, 0, 0))
            img.save(folder / f"tile_{i}.png")

        for i in range(2):
            img = Image.new('RGB', (90, 90), (0, i * 50, 0))
            img.save(folder / f"tile_{i}.jpg")

        # Load only PNG files
        db = TileDatabase()
        count = db.load_tiles_from_folder(str(folder), extensions=['.png'])

        assert count == 3

    def test_load_from_nonexistent_folder(self):
        """Test loading from non-existent folder."""
        db = TileDatabase()

        with pytest.raises(FileNotFoundError):
            db.load_tiles_from_folder("/nonexistent/folder")

    def test_load_from_file_not_folder(self, tmp_path):
        """Test loading from a file instead of folder."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("not a folder")

        db = TileDatabase()

        with pytest.raises(ValueError, match="not a directory"):
            db.load_tiles_from_folder(str(file_path))

    def test_load_empty_folder(self, tmp_path):
        """Test loading from empty folder."""
        folder = tmp_path / "empty"
        folder.mkdir()

        db = TileDatabase()
        count = db.load_tiles_from_folder(str(folder))

        assert count == 0
        assert len(db) == 0

    def test_load_folder_with_non_image_files(self, tmp_path):
        """Test loading folder with mixed file types."""
        folder = tmp_path / "mixed"
        folder.mkdir()

        # Create some images
        for i in range(3):
            img = Image.new('RGB', (90, 90), (i * 50, 0, 0))
            img.save(folder / f"tile_{i}.png")

        # Create non-image files
        (folder / "readme.txt").write_text("text file")
        (folder / "data.json").write_text("{}")

        db = TileDatabase()
        count = db.load_tiles_from_folder(str(folder))

        # Should only load the 3 image files
        assert count == 3


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        sections = []
        for i in range(9):
            bbox = BoundingBox(x=0, y=0, width=30, height=30)
            color = ColorAverage(r=255, g=0, b=0)
            sections.append(SectionData(bbox, color))

        tile_data = TileData("test.jpg", sections)
        result = SearchResult(tile_data, 42.5)

        assert result.tile_data == tile_data
        assert result.distance == 42.5

    def test_search_result_repr(self):
        """Test SearchResult string representation."""
        sections = []
        for i in range(9):
            bbox = BoundingBox(x=0, y=0, width=30, height=30)
            color = ColorAverage(r=255, g=0, b=0)
            sections.append(SectionData(bbox, color))

        tile_data = TileData("/path/to/test.jpg", sections)
        result = SearchResult(tile_data, 42.5)

        repr_str = repr(result)
        assert "test.jpg" in repr_str
        assert "42.5" in repr_str


class TestDistanceCalculation:
    """Tests for color distance calculation accuracy."""

    def create_solid_color_tile(self, path, color):
        """Helper to create a solid color test image."""
        img = Image.new('RGB', (90, 90), color)
        img.save(path)

    def test_identical_colors_have_zero_distance(self, tmp_path):
        """Test that identical colors have distance ~0."""
        db = TileDatabase()

        img_path = tmp_path / "test.png"
        self.create_solid_color_tile(img_path, (128, 128, 128))
        db.add_tile_from_path(str(img_path))
        db.build_index()

        # Query with same color
        query = [ColorAverage(r=128, g=128, b=128) for _ in range(9)]
        results = db.find_k_nearest_neighbors(query, k=1)

        assert len(results) == 1
        assert results[0].distance < 0.01  # Very close to 0

    def test_distance_ordering(self, tmp_path):
        """Test that distances are correctly ordered."""
        db = TileDatabase()

        # Create tiles with different distances from black
        colors = [(0, 0, 0), (50, 50, 50), (100, 100, 100), (200, 200, 200)]
        for i, color in enumerate(colors):
            img_path = tmp_path / f"test_{i}.png"
            self.create_solid_color_tile(img_path, color)
            db.add_tile_from_path(str(img_path))

        db.build_index()

        # Query for black
        query = [ColorAverage(r=0, g=0, b=0) for _ in range(9)]
        results = db.find_k_nearest_neighbors(query, k=4)

        # Distances should increase
        for i in range(len(results) - 1):
            assert results[i].distance <= results[i + 1].distance


class TestDifferentAlgorithms:
    """Tests for different k-NN algorithms."""

    def create_solid_color_tile(self, path, color):
        """Helper to create a solid color test image."""
        img = Image.new('RGB', (90, 90), color)
        img.save(path)

    @pytest.mark.parametrize("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
    def test_different_algorithms_same_results(self, tmp_path, algorithm):
        """Test that different algorithms produce same results."""
        db = TileDatabase(algorithm=algorithm)

        # Create test tiles
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, color in enumerate(colors):
            img_path = tmp_path / f"test_{i}.png"
            self.create_solid_color_tile(img_path, color)
            db.add_tile_from_path(str(img_path))

        db.build_index()

        # Query
        query = [ColorAverage(r=255, g=0, b=0) for _ in range(9)]
        result = db.find_nearest_neighbor(query)

        assert result is not None
        # Should match red tile
        colors = result.get_all_colors()
        assert all(c.r == 255 for c in colors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
