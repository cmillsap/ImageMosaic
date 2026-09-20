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

if OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover - exercised only on installs without OpenCV
    cv2 = None


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
    def test_cascade_actually_loads(self):
        """The Haar cascade must load, not silently fall back to no-op detection.

        Regression guard: detect_faces() swallows every failure and returns [],
        so an unloadable cascade is indistinguishable from "no faces present"
        unless initialization is asserted directly.
        """
        detector = FaceDetector()
        assert detector._ensure_initialized() is True, (
            f"Haar cascade failed to load: {detector._init_error}"
        )
        assert detector._cascade is not None
        assert not detector._cascade.empty()

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_returns_crop_regions(self, temp_dir):
        """Detected faces should be CropRegion objects."""
        detector = FaceDetector()
        if not detector._ensure_initialized():
            pytest.fail(f"Haar cascade not available: {detector._init_error}")

        img = Image.new('RGB', (200, 200), color=(128, 128, 128))
        faces = detector.detect_faces(img)
        assert isinstance(faces, list)
        assert all(isinstance(f, CropRegion) for f in faces)


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
    def test_saliency_backend_is_available(self):
        """cv2.saliency must exist, i.e. opencv-CONTRIB-python is installed.

        Regression guard: the base `opencv-python` package does not ship the
        saliency module, so SaliencyDetector silently degrades to returning
        None for every image and all cropping falls back to center-crop.
        """
        assert hasattr(cv2, 'saliency'), (
            "cv2.saliency missing - install opencv-contrib-python, "
            "not opencv-python (see requirements.txt)"
        )
        detector = SaliencyDetector()
        assert detector._ensure_initialized() is True, (
            f"Saliency detector failed to initialize: {detector._init_error}"
        )

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_detects_region_in_high_contrast_image(self, temp_dir):
        """Should locate a real salient region in a high-contrast image."""
        # Black image with a bright square at (80,80)-(120,120)
        img = Image.new('RGB', (200, 200), color=(0, 0, 0))
        for x in range(80, 120):
            for y in range(80, 120):
                img.putpixel((x, y), (255, 255, 255))

        detector = SaliencyDetector()
        if not detector._ensure_initialized():
            pytest.fail(f"Saliency unavailable: {detector._init_error}")

        result = detector.detect_salient_region(img)

        assert isinstance(result, CropRegion), (
            "Saliency returned no region for an unambiguous high-contrast target"
        )
        # The detected region must overlap the bright square, not sit elsewhere.
        assert 70 <= result.center_x <= 130, f"center_x off target: {result.center_x}"
        assert 70 <= result.center_y <= 130, f"center_y off target: {result.center_y}"
        assert result.confidence > 0.0

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

    def test_output_is_exactly_target_size(self, large_image):
        """Cropping alone left tiles at an arbitrary resolution, so cache
        entries were ~46x larger than needed (5.3 GB for a 10k library) and
        every consumer had to resize again."""
        preprocessor = TilePreprocessor(PreprocessorConfig(
            target_width=64, target_height=48,
            enable_face_detection=False, enable_saliency=False))
        assert preprocessor.preprocess_image(large_image).size == (64, 48)

    def test_cached_entry_is_stored_at_tile_size(self, temp_dir, large_image):
        """What lands on disk must be the final tile, not the full crop."""
        cache_dir = os.path.join(temp_dir, 'cache')
        preprocessor = TilePreprocessor(PreprocessorConfig(
            target_width=32, target_height=32, cache_dir=cache_dir,
            enable_face_detection=False, enable_saliency=False))
        preprocessor.preprocess_image(large_image)

        entries = [f for f in os.listdir(cache_dir) if f.endswith('.png')]
        assert len(entries) == 1
        with Image.open(os.path.join(cache_dir, entries[0])) as cached:
            assert cached.size == (32, 32)

    def test_cached_and_fresh_results_are_identical(self, temp_dir, large_image):
        """Resizing before both analysis and caching is what makes a tile's
        colour signature the same on a cold and a warm run."""
        cache_dir = os.path.join(temp_dir, 'cache')
        config = dict(target_width=40, target_height=40, cache_dir=cache_dir,
                      enable_face_detection=False, enable_saliency=False)

        fresh = TilePreprocessor(PreprocessorConfig(**config)).preprocess_image(large_image)
        warm = TilePreprocessor(PreprocessorConfig(**config)).preprocess_image(large_image)

        assert fresh.size == warm.size == (40, 40)
        assert list(fresh.getdata()) == list(warm.getdata())

    def test_select_region_single(self):
        """Single region should be returned unchanged."""
        config = PreprocessorConfig()
        preprocessor = TilePreprocessor(config)

        region = CropRegion(x=100, y=50, width=200, height=100, confidence=0.8)
        selected = preprocessor._select_region([region])

        assert selected is region

    def test_select_region_picks_highest_confidence(self):
        """The most confident candidate wins, regardless of position or size."""
        preprocessor = TilePreprocessor(PreprocessorConfig())

        weak_but_large = CropRegion(x=0, y=0, width=500, height=500,
                                    confidence=1.9)
        strong = CropRegion(x=300, y=300, width=120, height=120,
                            confidence=8.2)
        regions = [weak_but_large, strong,
                   CropRegion(x=900, y=50, width=80, height=80, confidence=3.9)]

        assert preprocessor._select_region(regions) is strong

    def test_select_region_breaks_ties_on_area(self):
        """With equal confidence (e.g. a detector with no real score) the
        largest region wins, keeping the choice deterministic."""
        preprocessor = TilePreprocessor(PreprocessorConfig())

        small = CropRegion(x=0, y=0, width=44, height=44, confidence=1.0)
        large = CropRegion(x=500, y=500, width=526, height=526, confidence=1.0)

        assert preprocessor._select_region([small, large]) is large
        assert preprocessor._select_region([large, small]) is large

    def test_select_region_does_not_union(self):
        """Regression: scattered candidates must not produce a region
        spanning the gaps between them, which collapses to a centre crop."""
        preprocessor = TilePreprocessor(PreprocessorConfig())

        regions = [
            CropRegion(x=100, y=100, width=50, height=50, confidence=2.0),
            CropRegion(x=3000, y=3000, width=50, height=50, confidence=5.0),
        ]
        selected = preprocessor._select_region(regions)

        assert selected.width == 50 and selected.height == 50
        assert (selected.x, selected.y) == (3000, 3000)

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


# ============================================================================
# Real-photograph detection tests
# ============================================================================
#
# Haar cascades are trained on photographs and do not fire on synthetic
# shapes, so the only way to prove face detection actually works is to run it
# against real images. These fixtures are large binaries and are not committed
# to the repository; drop the two files in the project root to enable the
# tests, otherwise they skip.

FACE_PHOTO = os.path.join(os.path.dirname(__file__), 'face.jpg')
NOFACE_PHOTO = os.path.join(os.path.dirname(__file__), 'noface.tif')

requires_face_photo = pytest.mark.skipif(
    not os.path.exists(FACE_PHOTO),
    reason="face.jpg fixture not present in project root"
)
requires_noface_photo = pytest.mark.skipif(
    not os.path.exists(NOFACE_PHOTO),
    reason="noface.tif fixture not present in project root"
)


def _pipeline_image(path):
    """Load an image and downscale it exactly as preprocess_image() does."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return TilePreprocessor(PreprocessorConfig())._rescale_if_needed(img)


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
class TestRealPhotoDetection:
    """End-to-end detection against real photographs."""

    @requires_face_photo
    def test_finds_a_face_in_a_real_photo(self):
        """The whole point of the feature: a real face must be detected."""
        faces = FaceDetector().detect_faces(_pipeline_image(FACE_PHOTO))
        assert len(faces) >= 1, "no face found in a photograph containing one"
        assert all(isinstance(f, CropRegion) for f in faces)
        assert all(f.width > 0 and f.height > 0 for f in faces)

    @requires_noface_photo
    def test_no_face_in_a_photo_without_one(self):
        """Guards against a detector that fires indiscriminately."""
        faces = FaceDetector().detect_faces(_pipeline_image(NOFACE_PHOTO))
        assert faces == [], f"false positives on a face-free photo: {faces}"

    @requires_face_photo
    def test_face_shifts_the_crop_away_from_centre(self):
        """A detected face must actually change where the tile is cropped.

        If subject detection silently degraded (as it did when the base
        opencv-python package was installed) the crop would be identical to
        a plain centre crop and this test would fail.
        """
        pre = TilePreprocessor(PreprocessorConfig(target_width=100,
                                                  target_height=100))
        work = _pipeline_image(FACE_PHOTO)

        region = pre._detect_region_of_interest(work)
        assert region is not None

        subject_crop = pre._crop_calculator.calculate_crop(work.size, region)
        centre_crop = pre._crop_calculator.calculate_crop(work.size, None)
        assert subject_crop != centre_crop, (
            "subject-aware crop is identical to a centre crop"
        )

    @requires_face_photo
    def test_detected_face_survives_into_the_crop(self):
        """The cropped tile must still contain the face it was centred on."""
        pre = TilePreprocessor(PreprocessorConfig(target_width=100,
                                                  target_height=100))
        work = _pipeline_image(FACE_PHOTO)

        face = FaceDetector().detect_faces(work)[0]
        left, top, right, bottom = pre._crop_calculator.calculate_crop(
            work.size, pre._detect_region_of_interest(work)
        )
        assert left <= face.center_x <= right
        assert top <= face.center_y <= bottom

    @requires_face_photo
    def test_false_positives_do_not_swamp_the_real_face(self):
        """At full resolution the cascade returns the real face plus several
        spurious matches. Selection must land on the real one.

        Regression: unioning every candidate produced a region spanning 83%
        of the frame, which is indistinguishable from a centre crop. The
        default pipeline hides this by downscaling first, so this test
        deliberately disables the rescale to exercise the multi-match path.
        """
        pre = TilePreprocessor(PreprocessorConfig(
            target_width=100, target_height=100,
            max_dimension_before_rescale=5000
        ))
        img = Image.open(FACE_PHOTO).convert('RGB')
        work = pre._rescale_if_needed(img)

        faces = pre._face_detector.detect_faces(work)
        assert len(faces) > 1, "expected several candidates at full resolution"

        roi = pre._select_region(faces)

        # The chosen region must be one of the candidates, not a union.
        assert any(roi.x == f.x and roi.y == f.y and roi.width == f.width
                   for f in faces)
        # It must be the most confident one.
        assert roi.confidence == max(f.confidence for f in faces)
        # And it must be a small part of the frame, not most of it.
        coverage = (roi.width * roi.height) / (work.size[0] * work.size[1])
        assert coverage < 0.25, f"selected region covers {coverage:.1%} of the frame"

    @requires_face_photo
    def test_face_confidences_are_not_all_equal(self):
        """Haar levelWeights must reach CropRegion, otherwise every candidate
        ties and highest-confidence selection degenerates to picking first."""
        pre = TilePreprocessor(PreprocessorConfig(
            max_dimension_before_rescale=5000))
        work = pre._rescale_if_needed(Image.open(FACE_PHOTO).convert('RGB'))

        confidences = [f.confidence for f in pre._face_detector.detect_faces(work)]
        assert len(set(confidences)) > 1, (
            f"all candidates share one confidence value: {confidences}"
        )

    @requires_face_photo
    def test_preprocess_image_end_to_end(self):
        """A real photo should come out at the configured aspect ratio."""
        pre = TilePreprocessor(PreprocessorConfig(target_width=120,
                                                  target_height=80))
        result = pre.preprocess_image(FACE_PHOTO)
        assert result.mode == 'RGB'
        assert result.size[0] / result.size[1] == pytest.approx(1.5, abs=0.01)
