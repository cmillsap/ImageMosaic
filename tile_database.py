"""
Tile Database Module

This module provides a database for storing and searching tile images
using k-nearest neighbor algorithm based on color similarity.
Uses scikit-learn's optimized NearestNeighbors implementation for performance.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from tile_analyzer import TileData, ImageTileAnalyzer, ColorAverage

if TYPE_CHECKING:
    from tile_preprocessor import TilePreprocessor


@dataclass
class SearchResult:
    """Represents a search result with the tile and its distance."""
    tile_data: TileData
    distance: float

    def __repr__(self):
        return f"SearchResult(image={os.path.basename(self.tile_data.image_path)}, distance={self.distance:.2f})"


class TileDatabase:
    """
    Database for storing and searching tile images using k-nearest neighbor.

    Uses scikit-learn's NearestNeighbors with efficient spatial indexing
    for fast similarity search based on 9-section color averages.
    """

    def __init__(self, algorithm: str = 'auto', metric: str = 'euclidean',
                 preprocessor: Optional['TilePreprocessor'] = None):
        """
        Initialize an empty tile database.

        Args:
            algorithm: Algorithm for nearest neighbor search.
                      Options: 'auto', 'ball_tree', 'kd_tree', 'brute'
                      'auto' will choose the best algorithm based on data
            metric: Distance metric to use. Default is 'euclidean'.
                   Other options: 'manhattan', 'chebyshev', 'minkowski'
            preprocessor: Optional TilePreprocessor for cropping/resizing tiles
                         before analysis. If None, tiles are analyzed as-is.
        """
        self._tiles: List[TileData] = []
        self._analyzer = ImageTileAnalyzer()
        self._knn_model: Optional[NearestNeighbors] = None
        self._feature_matrix: Optional[np.ndarray] = None
        self._algorithm = algorithm
        self._metric = metric
        self._is_fitted = False
        self._preprocessor = preprocessor

    def add_tile(self, tile_data: TileData) -> None:
        """
        Add a tile to the database.

        Note: After adding tiles, you must call build_index() before searching.

        Args:
            tile_data: TileData object to add
        """
        self._tiles.append(tile_data)
        self._is_fitted = False  # Mark as needing rebuild

    def add_tile_from_path(self, image_path: str) -> None:
        """
        Analyze and add a tile image from a file path.

        Note: After adding tiles, you must call build_index() before searching.

        Args:
            image_path: Path to the image file

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be analyzed
        """
        if self._preprocessor:
            processed_img = self._preprocessor.preprocess_image(image_path)
            tile_data = self._analyzer.analyze_pil_image(processed_img, image_path)
        else:
            tile_data = self._analyzer.analyze_image(image_path)
        self.add_tile(tile_data)

    def load_tiles_from_folder(self, folder_path: str,
                               extensions: Optional[List[str]] = None,
                               auto_build_index: bool = True,
                               recursive: bool = False) -> int:
        """
        Load all tile images from a folder.

        Args:
            folder_path: Path to folder containing tile images
            extensions: List of file extensions to include (e.g., ['.png', '.jpg'])
                       If None, defaults to common image formats
            auto_build_index: If True, automatically builds the search index after loading
            recursive: If True, also scan subdirectories for tile images

        Returns:
            Number of tiles successfully loaded

        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

        # Convert extensions to lowercase for case-insensitive matching
        extensions = [ext.lower() for ext in extensions]

        loaded_count = 0
        failed_files = []

        if recursive:
            # Walk through all subdirectories
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    file_path = os.path.join(root, filename)

                    # Check file extension
                    _, ext = os.path.splitext(filename)
                    if ext.lower() not in extensions:
                        continue

                    try:
                        self.add_tile_from_path(file_path)
                        loaded_count += 1
                    except Exception as e:
                        failed_files.append((file_path, str(e)))
        else:
            # Only scan the top-level folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # Skip if not a file
                if not os.path.isfile(file_path):
                    continue

                # Check file extension
                _, ext = os.path.splitext(filename)
                if ext.lower() not in extensions:
                    continue

                try:
                    self.add_tile_from_path(file_path)
                    loaded_count += 1
                except Exception as e:
                    failed_files.append((filename, str(e)))

        # Build index if requested and tiles were loaded
        if auto_build_index and loaded_count > 0:
            self.build_index()

        return loaded_count

    def build_index(self) -> None:
        """
        Build the k-NN search index from loaded tiles.

        This must be called after adding tiles and before searching.
        Uses scikit-learn's NearestNeighbors for efficient indexing.

        Raises:
            ValueError: If no tiles have been added
        """
        if not self._tiles:
            raise ValueError("Cannot build index: no tiles in database")

        # Convert tiles to feature vectors (9 colors × 3 RGB = 27 dimensions)
        self._feature_matrix = self._tiles_to_feature_matrix(self._tiles)

        # Create and fit the k-NN model
        self._knn_model = NearestNeighbors(
            algorithm=self._algorithm,
            metric=self._metric
        )
        self._knn_model.fit(self._feature_matrix)
        self._is_fitted = True

    def _tiles_to_feature_matrix(self, tiles: List[TileData]) -> np.ndarray:
        """
        Convert list of TileData to a feature matrix.

        Each tile is represented as a 27-dimensional vector:
        [r0, g0, b0, r1, g1, b1, ..., r8, g8, b8]

        Args:
            tiles: List of TileData objects

        Returns:
            NumPy array of shape (n_tiles, 27)
        """
        n_tiles = len(tiles)
        features = np.zeros((n_tiles, 27), dtype=np.float32)

        for i, tile in enumerate(tiles):
            colors = tile.get_all_colors()
            for j, color in enumerate(colors):
                features[i, j * 3] = color.r
                features[i, j * 3 + 1] = color.g
                features[i, j * 3 + 2] = color.b

        return features

    def _colors_to_feature_vector(self, colors: List[ColorAverage]) -> np.ndarray:
        """
        Convert a list of 9 ColorAverage objects to a feature vector.

        Args:
            colors: List of 9 ColorAverage objects

        Returns:
            NumPy array of shape (1, 27)
        """
        if len(colors) != 9:
            raise ValueError(f"Expected 9 colors, got {len(colors)}")

        features = np.zeros((1, 27), dtype=np.float32)
        for i, color in enumerate(colors):
            features[0, i * 3] = color.r
            features[0, i * 3 + 1] = color.g
            features[0, i * 3 + 2] = color.b

        return features

    def size(self) -> int:
        """Return the number of tiles in the database."""
        return len(self._tiles)

    def clear(self) -> None:
        """Remove all tiles from the database."""
        self._tiles.clear()
        self._knn_model = None
        self._feature_matrix = None
        self._is_fitted = False

    def is_ready(self) -> bool:
        """
        Check if the database is ready for searches.

        Returns:
            True if index has been built, False otherwise
        """
        return self._is_fitted

    def find_nearest_neighbor(self, query_colors: List[ColorAverage]) -> Optional[TileData]:
        """
        Find the single nearest neighbor tile.

        Args:
            query_colors: List of 9 ColorAverage objects to match

        Returns:
            The nearest matching TileData, or None if database is empty

        Raises:
            ValueError: If query_colors doesn't contain exactly 9 colors
            RuntimeError: If index hasn't been built yet
        """
        results = self.find_k_nearest_neighbors(query_colors, k=1)
        return results[0].tile_data if results else None

    def find_k_nearest_neighbors(self, query_colors: List[ColorAverage],
                                 k: int = 1) -> List[SearchResult]:
        """
        Find the k nearest neighbor tiles using color similarity.

        Args:
            query_colors: List of 9 ColorAverage objects to match
            k: Number of nearest neighbors to return

        Returns:
            List of SearchResult objects, sorted by distance (closest first)

        Raises:
            ValueError: If query_colors doesn't contain exactly 9 colors
            ValueError: If k is less than 1
            RuntimeError: If index hasn't been built yet
        """
        if len(query_colors) != 9:
            raise ValueError(f"Expected 9 colors, got {len(query_colors)}")

        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

        if not self._is_fitted:
            raise RuntimeError("Index not built. Call build_index() before searching.")

        if not self._tiles:
            return []

        # Convert query to feature vector
        query_vector = self._colors_to_feature_vector(query_colors)

        # Perform k-NN search
        # kneighbors returns (distances, indices)
        k_actual = min(k, len(self._tiles))
        distances, indices = self._knn_model.kneighbors(query_vector, n_neighbors=k_actual)

        # Build result list
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append(SearchResult(self._tiles[idx], float(dist)))

        return results

    def get_all_tiles(self) -> List[TileData]:
        """
        Get all tiles in the database.

        Returns:
            List of all TileData objects
        """
        return self._tiles.copy()

    def get_tile_by_index(self, index: int) -> TileData:
        """
        Get a tile by its index in the database.

        Args:
            index: Index of the tile (0-based)

        Returns:
            TileData at the specified index

        Raises:
            IndexError: If index is out of bounds
        """
        return self._tiles[index]

    def __len__(self) -> int:
        """Return the number of tiles in the database."""
        return len(self._tiles)

    def __repr__(self):
        status = "ready" if self._is_fitted else "not indexed"
        return f"TileDatabase(tiles={len(self._tiles)}, {status})"
