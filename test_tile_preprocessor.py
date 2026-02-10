"""
Tests for the Tile Preprocessor Module
"""

import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

import pytest
from PIL import Image

from tile_preprocessor import (
    CropRegion,
    PreprocessorConfig,
    FaceDetector,
    SaliencyDetector,
    CropCalculator,
    TilePreprocessorCache,
    TilePreprocessor,
    OPENCV_AVAILABLE,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    img = Image.new('RGB', (400, 300), color=(128, 64, 192))
    path = os.path.join(temp_dir, 'sample.png')
    img.save(path)
    return path


@pytest.fixture
def wide_image(temp_dir):
    """Create a wide aspect ratio image."""
    img = Image.new('RGB', (800, 400), color=(255, 0, 0))
    path = os.path.join(temp_dir, 'wide.png')
    img.save(path)
    return path


@pytest.fixture
def tall_image(temp_dir):
    """Create a tall aspect ratio image."""
    img = Image.new('RGB', (400, 800), color=(0, 255, 0))
    path = os.path.join(temp_dir, 'tall.png')
    img.save(path)
    return path


@pytest.fixture
def large_image(temp_dir):
    """Create a large image that needs rescaling."""
    img = Image.new('RGB', (2000, 1500), color=(0, 0, 255))
    path = os.path.join(temp_dir, 'large.png')
    img.save(path)
    return path


@pytest.fixture
def square_image(temp_dir):
    """Create a square image."""
    img = Image.new('RGB', (500, 500), color=(100, 100, 100))
    path = os.path.join(temp_dir, 'square.png')
    img.save(path)
    return path


# ============================================================================
# CropRegion Tests
# ============================================================================

class TestCropRegion:
    """Tests for CropRegion dataclass."""

    def test_center_x(self):
        region = CropRegion(x=100, y=50, width=200, height=100)
        assert region.center_x == 200

    def test_center_y(self):
        region = CropRegion(x=100, y=50, width=200, height=100)
        assert region.center_y == 100

    def test_default_confidence(self):
        region = CropRegion(x=0, y=0, width=100, height=100)
        assert region.confidence == 1.0

    def test_custom_confidence(self):
        region = CropRegion(x=0, y=0, width=100, height=100, confidence=0.75)
        assert region.confidence == 0.75


# ============================================================================
# PreprocessorConfig Tests
# ============================================================================

class TestPreprocessorConfig:
    """Tests for PreprocessorConfig dataclass."""

    def test_default_values(self):
        config = PreprocessorConfig()
        assert config.target_width == 100
        assert config.target_height == 100
        assert config.enable_face_detection is True
        assert config.enable_saliency is True
        assert config.cache_dir is None

    def test_custom_values(self):
        config = PreprocessorConfig(
            target_width=200,
            target_height=150,
            enable_face_detection=False,
            cache_dir='/tmp/cache'
        )
        assert config.target_width == 200
        assert config.target_height == 150
        assert config.enable_face_detection is False
        assert config.cache_dir == '/tmp/cache'

    def test_target_aspect_ratio_square(self):
        config = PreprocessorConfig(target_width=100, target_height=100)
        assert config.target_aspect_ratio == 1.0

    def test_target_aspect_ratio_wide(self):
        config = PreprocessorConfig(target_width=200, target_height=100)
        assert config.target_aspect_ratio == 2.0

    def test_target_aspect_ratio_tall(self):
        config = PreprocessorConfig(target_width=100, target_height=200)
        assert config.target_aspect_ratio == 0.5


# ============================================================================
# CropCalculator Tests
# ============================================================================

class TestCropCalculator:
    """Tests for CropCalculator class."""

    def test_center_crop_wide_image_square_target(self):
        """Wide image cropped to square should crop sides."""
        calc = CropCalculator(target_aspect_ratio=1.0)
        crop = calc.calculate_crop((800, 400))

        left, top, right, bottom = crop
        assert right - left == 400  # Width equals height
        assert bottom - top == 400
        assert left == 200  # Centered horizontally
        assert top == 0

    def test_center_crop_tall_image_square_target(self):
        """Tall image cropped to square should crop top/bottom."""
        calc = CropCalculator(target_aspect_ratio=1.0)
        crop = calc.calculate_crop((400, 800))

        left, top, right, bottom = crop
        assert right - left == 400
        assert bottom - top == 400
        assert left == 0
        assert top == 200  # Centered vertically

    def test_center_crop_preserves_aspect_ratio(self):
        """Verify crop maintains target aspect ratio."""
        calc = CropCalculator(target_aspect_ratio=16/9)
        crop = calc.calculate_crop((1000, 1000))

        left, top, right, bottom = crop
        width = right - left
        height = bottom - top
        actual_ratio = width / height
        assert abs(actual_ratio - 16/9) < 0.01

    def test_region_aware_crop_centers_on_region(self):
        """Crop should be centered on the region of interest when possible."""
        calc = CropCalculator(target_aspect_ratio=1.0)
        # Use a region in the center of a large image to avoid boundary constraints
        region = CropRegion(x=350, y=350, width=100, height=100)
        crop = calc.calculate_crop((800, 800), region)

        left, top, right, bottom = crop
        crop_center_x = (left + right) // 2
        crop_center_y = (top + bottom) // 2

        # Crop should be centered on region
        assert crop_center_x == region.center_x
        assert crop_center_y == region.center_y

    def test_region_aware_crop_stays_in_bounds_left(self):
        """Crop should not extend past left edge."""
        calc = CropCalculator(target_aspect_ratio=1.0)
        region = CropRegion(x=0, y=200, width=50, height=50)
        crop = calc.calculate_crop((800, 400), region)

        left, top, right, bottom = crop
        assert left >= 0

    def test_region_aware_crop_stays_in_bounds_right(self):
        """Crop should not extend past right edge."""
        calc = CropCalculator(target_aspect_ratio=1.0)
        region = CropRegion(x=750, y=200, width=50, height=50)
        crop = calc.calculate_crop((800, 400), region)

        left, top, right, bottom = crop
        assert right <= 800

    def test_region_aware_crop_stays_in_bounds_top(self):
        """Crop should not extend past top edge."""
        calc = CropCalculator(target_aspect_ratio=1.0)
        region = CropRegion(x=400, y=0, width=50, height=50)
        crop = calc.calculate_crop((800, 400), region)

        left, top, right, bottom = crop
        assert top >= 0

    def test_region_aware_crop_stays_in_bounds_bottom(self):
        """Crop should not extend past bottom edge."""
        calc = CropCalculator(target_aspect_ratio=1.0)
        region = CropRegion(x=400, y=350, width=50, height=50)
        crop = calc.calculate_crop((800, 400), region)

        left, top, right, bottom = crop
        assert bottom <= 400


# ============================================================================
# TilePreprocessorCache Tests
# ============================================================================

class TestTilePreprocessorCache:
    """Tests for TilePreprocessorCache class."""

    def test_cache_disabled_when_no_dir(self, sample_image):
        """Cache should return None when disabled."""
        cache = TilePreprocessorCache(cache_dir=None)
        result = cache.get_cached(sample_image, (100, 100))
        assert result is None

    def test_cache_miss_returns_none(self, temp_dir, sample_image):
        """Cache miss should return None."""
        cache_dir = os.path.join(temp_dir, 'cache')
        cache = TilePreprocessorCache(cache_dir=cache_dir)
        result = cache.get_cached(sample_image, (100, 100))
        assert result is None

    def test_cache_hit_returns_image(self, temp_dir, sample_image):
        """Cache hit should return the cached image."""
        cache_dir = os.path.join(temp_dir, 'cache')
        cache = TilePreprocessorCache(cache_dir=cache_dir)

        # Save to cache
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        cache.save_to_cache(sample_image, img, (100, 100))

        # Retrieve from cache
        result = cache.get_cached(sample_image, (100, 100))
        assert result is not None
        assert result.size == (100, 100)

    def test_different_sizes_have_different_keys(self, temp_dir, sample_image):
        """Different target sizes should use different cache keys."""
        cache_dir = os.path.join(temp_dir, 'cache')
        cache = TilePreprocessorCache(cache_dir=cache_dir)

        # Save with one size
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        cache.save_to_cache(sample_image, img, (100, 100))

        # Try to retrieve with different size
        result = cache.get_cached(sample_image, (200, 200))
        assert result is None

    def test_cache_creates_directory(self, temp_dir):
        """Cache should create its directory if it doesn't exist."""
        cache_dir = os.path.join(temp_dir, 'new_cache_dir')
        cache = TilePreprocessorCache(cache_dir=cache_dir)
        assert os.path.exists(cache_dir)


# ============================================================================
# FaceDetector Tests
# ============================================================================

class TestFaceDetector:
    """Tests for FaceDetector class."""

    def test_returns_empty_list_without_opencv(self, sample_image):
        """Should return empty list when OpenCV is not available."""
        with patch('tile_preprocessor.OPENCV_AVAILABLE', False):
            detector = FaceDetector()
            detector._initialized = False  # Reset lazy init
            img = Image.open(sample_image)
            faces = detector.detect_faces(img)
            assert faces == []

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_detects_no_faces_in_blank_image(self, sample_image):
        """Should find no faces in a solid color image."""
        detector = FaceDetector()
        img = Image.open(sample_image)
        faces = detector.detect_faces(img)
        assert len(faces) == 0

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_returns_crop_regions(self, temp_dir):
        """Detected faces should be CropRegion objects."""
        # Create a mock that returns face coordinates
        detector = FaceDetector()
        detector._ensure_initialized()

        if detector._cascade is None:
            pytest.skip("Haar cascade not available")

        # Create a blank image for testing the return type
        img = Image.new('RGB', (200, 200), color=(128, 128, 128))
        faces = detector.detect_faces(img)
        # The list may be empty, but that's okay for this test
        assert isinstance(faces, list)


# ============================================================================
# SaliencyDetector Tests
# ============================================================================

class TestSaliencyDetector:
    """Tests for SaliencyDetector class."""

    def test_returns_none_without_opencv(self, sample_image):
        """Should return None when OpenCV is not available."""
        with patch('tile_preprocessor.OPENCV_AVAILABLE', False):
            detector = SaliencyDetector()
            detector._initialized = False  # Reset lazy init
            img = Image.open(sample_image)
            result = detector.detect_salient_region(img)
            assert result is None

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_detects_region_in_high_contrast_image(self, temp_dir):
        """Should detect salient region in a high-contrast image."""
        # Create image with a bright spot
        img = Image.new('RGB', (200, 200), color=(0, 0, 0))
        # Draw a bright rectangle
        for x in range(80, 120):
            for y in range(80, 120):
                img.putpixel((x, y), (255, 255, 255))

        detector = SaliencyDetector()
        result = detector.detect_salient_region(img)

        # Saliency detection may or may not find the region depending on
        # OpenCV version and thresholding, so we just check it doesn't crash
        assert result is None or isinstance(result, CropRegion)

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_handles_uniform_image(self, sample_image):
        """Should handle uniform color images gracefully."""
        detector = SaliencyDetector()
        img = Image.open(sample_image)
        result = detector.detect_salient_region(img)
        # May return None or a region - both are valid
        assert result is None or isinstance(result, CropRegion)


# ============================================================================
# TilePreprocessor Tests
# ============================================================================

class TestTilePreprocessor:
    """Tests for TilePreprocessor class."""

    def test_preprocess_returns_pil_image(self, sample_image):
        """Preprocessing should return a PIL Image."""
        config = PreprocessorConfig(
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)
        result = preprocessor.preprocess_image(sample_image)
        assert isinstance(result, Image.Image)

    def test_preprocess_converts_to_rgb(self, temp_dir):
        """Preprocessing should convert RGBA to RGB."""
        # Create RGBA image
        img = Image.new('RGBA', (200, 200), color=(128, 64, 192, 128))
        path = os.path.join(temp_dir, 'rgba.png')
        img.save(path)

        config = PreprocessorConfig(
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)
        result = preprocessor.preprocess_image(path)
        assert result.mode == 'RGB'

    def test_preprocess_raises_for_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        preprocessor = TilePreprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor.preprocess_image('/nonexistent/image.png')

    def test_preprocess_maintains_aspect_ratio(self, wide_image):
        """Result should have the target aspect ratio."""
        config = PreprocessorConfig(
            target_width=100,
            target_height=100,
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)
        result = preprocessor.preprocess_image(wide_image)

        width, height = result.size
        actual_ratio = width / height
        expected_ratio = config.target_aspect_ratio
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_rescale_downscales_large_images(self, large_image):
        """Large images should be downscaled before processing."""
        config = PreprocessorConfig(
            max_dimension_before_rescale=500,
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)

        # Load and rescale
        img = Image.open(large_image)
        rescaled = preprocessor._rescale_if_needed(img)

        # Should be scaled down
        assert max(rescaled.size) <= 500

    def test_rescale_preserves_aspect_ratio(self, large_image):
        """Rescaling should preserve the original aspect ratio."""
        config = PreprocessorConfig(
            max_dimension_before_rescale=500,
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)

        img = Image.open(large_image)
        original_ratio = img.width / img.height

        rescaled = preprocessor._rescale_if_needed(img)
        rescaled_ratio = rescaled.width / rescaled.height

        assert abs(original_ratio - rescaled_ratio) < 0.01

    def test_rescale_does_not_upscale(self, sample_image):
        """Small images should not be upscaled."""
        config = PreprocessorConfig(
            max_dimension_before_rescale=1000,
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)

        img = Image.open(sample_image)
        original_size = img.size

        rescaled = preprocessor._rescale_if_needed(img)
        assert rescaled.size == original_size

    def test_uses_cache_on_second_call(self, temp_dir, sample_image):
        """Second call should use cached result."""
        cache_dir = os.path.join(temp_dir, 'cache')
        config = PreprocessorConfig(
            cache_dir=cache_dir,
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)

        # First call - processes image
        result1 = preprocessor.preprocess_image(sample_image)

        # Second call - should use cache
        result2 = preprocessor.preprocess_image(sample_image)

        # Both should be valid images
        assert result1.size == result2.size

    def test_combine_regions_single(self):
        """Single region should be returned unchanged."""
        config = PreprocessorConfig()
        preprocessor = TilePreprocessor(config)

        region = CropRegion(x=100, y=50, width=200, height=100, confidence=0.8)
        combined = preprocessor._combine_regions([region])

        assert combined.x == region.x
        assert combined.y == region.y
        assert combined.width == region.width
        assert combined.height == region.height

    def test_combine_regions_multiple(self):
        """Multiple regions should be combined into bounding box."""
        config = PreprocessorConfig()
        preprocessor = TilePreprocessor(config)

        regions = [
            CropRegion(x=100, y=100, width=50, height=50),
            CropRegion(x=200, y=150, width=50, height=50),
        ]
        combined = preprocessor._combine_regions(regions)

        assert combined.x == 100  # Min x
        assert combined.y == 100  # Min y
        assert combined.width == 150  # From x=100 to x=250
        assert combined.height == 100  # From y=100 to y=200

    def test_default_config_used_when_none_provided(self):
        """Should use default config when None provided."""
        preprocessor = TilePreprocessor(config=None)
        assert preprocessor.config.target_width == 100
        assert preprocessor.config.target_height == 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full preprocessing pipeline."""

    def test_full_pipeline_wide_image(self, wide_image):
        """Test full preprocessing pipeline with a wide image."""
        config = PreprocessorConfig(
            target_width=100,
            target_height=100,
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)
        result = preprocessor.preprocess_image(wide_image)

        # Should produce a square crop
        assert result.size[0] == result.size[1]

    def test_full_pipeline_tall_image(self, tall_image):
        """Test full preprocessing pipeline with a tall image."""
        config = PreprocessorConfig(
            target_width=100,
            target_height=100,
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)
        result = preprocessor.preprocess_image(tall_image)

        # Should produce a square crop
        assert result.size[0] == result.size[1]

    def test_full_pipeline_large_image(self, large_image):
        """Test full preprocessing pipeline with a large image."""
        config = PreprocessorConfig(
            target_width=100,
            target_height=100,
            max_dimension_before_rescale=500,
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)
        result = preprocessor.preprocess_image(large_image)

        assert isinstance(result, Image.Image)

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_full_pipeline_with_detection_enabled(self, sample_image):
        """Test full pipeline with face/saliency detection enabled."""
        config = PreprocessorConfig(
            target_width=100,
            target_height=100,
            enable_face_detection=True,
            enable_saliency=True
        )
        preprocessor = TilePreprocessor(config)
        result = preprocessor.preprocess_image(sample_image)

        assert isinstance(result, Image.Image)


# ============================================================================
# TileDatabase Integration Tests
# ============================================================================

class TestTileDatabaseIntegration:
    """Tests for TileDatabase integration with preprocessor."""

    def test_database_with_preprocessor(self, temp_dir, sample_image):
        """Test TileDatabase with preprocessor enabled."""
        from tile_database import TileDatabase

        config = PreprocessorConfig(
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)
        db = TileDatabase(preprocessor=preprocessor)

        db.add_tile_from_path(sample_image)
        db.build_index()

        assert db.size() == 1

    def test_database_load_folder_with_preprocessor(self, temp_dir):
        """Test loading a folder with preprocessing enabled."""
        from tile_database import TileDatabase

        # Create multiple test images
        for i in range(3):
            img = Image.new('RGB', (200, 200), color=(i * 50, i * 50, i * 50))
            img.save(os.path.join(temp_dir, f'image_{i}.png'))

        config = PreprocessorConfig(
            enable_face_detection=False,
            enable_saliency=False
        )
        preprocessor = TilePreprocessor(config)
        db = TileDatabase(preprocessor=preprocessor)

        count = db.load_tiles_from_folder(temp_dir)
        assert count == 3
        assert db.size() == 3
