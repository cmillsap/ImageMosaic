"""
Tile Preprocessor Module

This module provides intelligent preprocessing to crop tile images to a uniform size
with subject-aware cropping (preserving faces and salient regions).
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

# Optional OpenCV import for face/saliency detection
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    np = None

logger = logging.getLogger(__name__)


@dataclass
class CropRegion:
    """Represents a region of interest for cropping."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def center_x(self) -> int:
        """Get the x-coordinate of the region center."""
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """Get the y-coordinate of the region center."""
        return self.y + self.height // 2

    def __repr__(self):
        return f"CropRegion(x={self.x}, y={self.y}, w={self.width}, h={self.height}, conf={self.confidence:.2f})"


@dataclass
class PreprocessorConfig:
    """Configuration for the tile preprocessor."""
    target_width: int = 100
    target_height: int = 100
    enable_face_detection: bool = True
    enable_saliency: bool = True
    cache_dir: Optional[str] = None
    max_dimension_before_rescale: int = 1000

    @property
    def target_aspect_ratio(self) -> float:
        """Get the target aspect ratio (width / height)."""
        return self.target_width / self.target_height


class FaceDetector:
    """Detects faces in images using OpenCV Haar cascade."""

    def __init__(self):
        """Initialize the face detector with Haar cascade classifier."""
        self._cascade = None
        self._initialized = False
        self._init_error = None

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the cascade classifier."""
        if self._initialized:
            return self._cascade is not None

        self._initialized = True

        if not OPENCV_AVAILABLE:
            self._init_error = "OpenCV not available"
            return False

        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                self._init_error = "Failed to load Haar cascade"
                self._cascade = None
                return False
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"Failed to initialize face detector: {e}")
            return False

    def detect_faces(self, image: Image.Image) -> List[CropRegion]:
        """
        Detect faces in an image.

        Args:
            image: PIL Image to analyze

        Returns:
            List of CropRegion objects for detected faces
        """
        if not self._ensure_initialized():
            return []

        try:
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            regions = []
            for (x, y, w, h) in faces:
                regions.append(CropRegion(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    confidence=1.0
                ))

            return regions

        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []


class SaliencyDetector:
    """Detects salient regions in images using OpenCV spectral residual saliency."""

    def __init__(self):
        """Initialize the saliency detector."""
        self._saliency = None
        self._initialized = False
        self._init_error = None

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the saliency detector."""
        if self._initialized:
            return self._saliency is not None

        self._initialized = True

        if not OPENCV_AVAILABLE:
            self._init_error = "OpenCV not available"
            return False

        try:
            self._saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"Failed to initialize saliency detector: {e}")
            return False

    def detect_salient_region(self, image: Image.Image) -> Optional[CropRegion]:
        """
        Detect the most salient region in an image.

        Args:
            image: PIL Image to analyze

        Returns:
            CropRegion for the salient area, or None if detection fails
        """
        if not self._ensure_initialized():
            return None

        try:
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Compute saliency map
            success, saliency_map = self._saliency.computeSaliency(cv_image)
            if not success:
                return None

            # Normalize and threshold the saliency map
            saliency_map = (saliency_map * 255).astype(np.uint8)
            _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours of salient regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Find the bounding box that encompasses all significant contours
            # Filter out very small contours (noise)
            min_area = (image.width * image.height) * 0.01  # At least 1% of image
            significant_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

            if not significant_contours:
                # Use the largest contour if no significant ones found
                significant_contours = [max(contours, key=cv2.contourArea)]

            # Get bounding box for all significant contours
            all_points = np.vstack(significant_contours)
            x, y, w, h = cv2.boundingRect(all_points)

            # Calculate confidence based on saliency strength
            region_saliency = saliency_map[y:y+h, x:x+w]
            confidence = float(np.mean(region_saliency) / 255.0)

            return CropRegion(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                confidence=confidence
            )

        except Exception as e:
            logger.warning(f"Saliency detection failed: {e}")
            return None


class CropCalculator:
    """Calculates optimal crop boundaries while preserving regions of interest."""

    def __init__(self, target_aspect_ratio: float):
        """
        Initialize the crop calculator.

        Args:
            target_aspect_ratio: Desired width/height ratio
        """
        self.target_aspect_ratio = target_aspect_ratio

    def calculate_crop(self, image_size: Tuple[int, int],
                       region_of_interest: Optional[CropRegion] = None) -> Tuple[int, int, int, int]:
        """
        Calculate the optimal crop box for an image.

        Args:
            image_size: Tuple of (width, height) of the source image
            region_of_interest: Optional CropRegion to center the crop around

        Returns:
            Tuple of (left, top, right, bottom) crop coordinates
        """
        width, height = image_size

        if region_of_interest:
            return self._region_aware_crop(image_size, region_of_interest)
        else:
            return self._center_crop(image_size)

    def _center_crop(self, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate a center crop that maintains the target aspect ratio.

        Args:
            image_size: Tuple of (width, height)

        Returns:
            Tuple of (left, top, right, bottom)
        """
        width, height = image_size
        image_aspect = width / height

        if image_aspect > self.target_aspect_ratio:
            # Image is wider than target - crop sides
            new_width = int(height * self.target_aspect_ratio)
            new_height = height
        else:
            # Image is taller than target - crop top/bottom
            new_width = width
            new_height = int(width / self.target_aspect_ratio)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        return (left, top, right, bottom)

    def _region_aware_crop(self, image_size: Tuple[int, int],
                           region: CropRegion) -> Tuple[int, int, int, int]:
        """
        Calculate a crop that includes the region of interest.

        Args:
            image_size: Tuple of (width, height)
            region: CropRegion to preserve in the crop

        Returns:
            Tuple of (left, top, right, bottom)
        """
        width, height = image_size
        image_aspect = width / height

        # Calculate crop dimensions that maintain aspect ratio
        if image_aspect > self.target_aspect_ratio:
            # Image is wider than target - crop sides
            crop_width = int(height * self.target_aspect_ratio)
            crop_height = height
        else:
            # Image is taller than target - crop top/bottom
            crop_width = width
            crop_height = int(width / self.target_aspect_ratio)

        # Center the crop on the region of interest
        center_x = region.center_x
        center_y = region.center_y

        left = center_x - crop_width // 2
        top = center_y - crop_height // 2

        # Ensure crop stays within image bounds
        if left < 0:
            left = 0
        elif left + crop_width > width:
            left = width - crop_width

        if top < 0:
            top = 0
        elif top + crop_height > height:
            top = height - crop_height

        right = left + crop_width
        bottom = top + crop_height

        return (left, top, right, bottom)


class TilePreprocessorCache:
    """Caches preprocessed tile images to avoid redundant processing."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for cache files. If None, caching is disabled.
        """
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, image_path: str, target_size: Tuple[int, int]) -> str:
        """
        Generate a unique cache key for an image and target size.

        Args:
            image_path: Path to the source image
            target_size: Tuple of (width, height)

        Returns:
            Hash string for cache lookup
        """
        abs_path = os.path.abspath(image_path)
        mtime = os.path.getmtime(abs_path)
        key_string = f"{abs_path}:{mtime}:{target_size[0]}x{target_size[1]}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self._cache_dir, f"{cache_key}.png")

    def get_cached(self, image_path: str, target_size: Tuple[int, int]) -> Optional[Image.Image]:
        """
        Retrieve a cached image if available.

        Args:
            image_path: Path to the original image
            target_size: Target dimensions (width, height)

        Returns:
            Cached PIL Image, or None if not cached
        """
        if not self._cache_dir:
            return None

        try:
            cache_key = self._get_cache_key(image_path, target_size)
            cache_path = self._get_cache_path(cache_key)

            if os.path.exists(cache_path):
                return Image.open(cache_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Cache read failed for {image_path}: {e}")

        return None

    def save_to_cache(self, image_path: str, image: Image.Image,
                      target_size: Tuple[int, int]) -> None:
        """
        Save a processed image to the cache.

        Args:
            image_path: Path to the original image
            image: Processed PIL Image to cache
            target_size: Target dimensions (width, height)
        """
        if not self._cache_dir:
            return

        try:
            cache_key = self._get_cache_key(image_path, target_size)
            cache_path = self._get_cache_path(cache_key)
            image.save(cache_path, 'PNG')
        except Exception as e:
            logger.warning(f"Cache write failed for {image_path}: {e}")


class TilePreprocessor:
    """
    Main entry point for tile preprocessing.

    Handles rescaling, subject detection (faces/saliency), and cropping
    to produce uniformly sized tile images.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        """
        Initialize the tile preprocessor.

        Args:
            config: PreprocessorConfig with target dimensions and options.
                   If None, uses default configuration.
        """
        self.config = config or PreprocessorConfig()
        self._face_detector = FaceDetector() if self.config.enable_face_detection else None
        self._saliency_detector = SaliencyDetector() if self.config.enable_saliency else None
        self._crop_calculator = CropCalculator(self.config.target_aspect_ratio)
        self._cache = TilePreprocessorCache(self.config.cache_dir)

    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Preprocess an image: rescale if needed, detect subject, and crop.

        Args:
            image_path: Path to the source image

        Returns:
            Preprocessed PIL Image cropped to target aspect ratio

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be processed
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        target_size = (self.config.target_width, self.config.target_height)

        # Check cache first
        cached = self._cache.get_cached(image_path, target_size)
        if cached is not None:
            return cached

        try:
            # Load and convert image
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Work with a copy since we may modify it
                working_img = img.copy()

            # Rescale if needed
            working_img = self._rescale_if_needed(working_img)

            # Detect region of interest
            region = self._detect_region_of_interest(working_img)

            # Apply crop
            result = self._apply_crop(working_img, region)

            # Save to cache
            self._cache.save_to_cache(image_path, result, target_size)

            return result

        except Exception as e:
            raise ValueError(f"Failed to preprocess image {image_path}: {str(e)}")

    def _rescale_if_needed(self, image: Image.Image) -> Image.Image:
        """
        Downscale image if it's larger than needed.

        Only downscales, never upscales. Scales so the smaller dimension
        matches the target, leaving room for cropping.

        Args:
            image: PIL Image to potentially rescale

        Returns:
            Rescaled PIL Image (or original if no rescaling needed)
        """
        width, height = image.size
        max_dim = self.config.max_dimension_before_rescale

        # Check if rescaling is needed
        if width <= max_dim and height <= max_dim:
            return image

        # Calculate scale factor to bring largest dimension down to max
        scale = max_dim / max(width, height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _detect_region_of_interest(self, image: Image.Image) -> Optional[CropRegion]:
        """
        Detect the primary region of interest in an image.

        Tries face detection first, then saliency detection.

        Args:
            image: PIL Image to analyze

        Returns:
            CropRegion for the detected subject, or None for center crop
        """
        # Try face detection first
        if self._face_detector:
            faces = self._face_detector.detect_faces(image)
            if faces:
                # Combine multiple faces into a single region
                return self._combine_regions(faces)

        # Fall back to saliency detection
        if self._saliency_detector:
            salient_region = self._saliency_detector.detect_salient_region(image)
            if salient_region:
                return salient_region

        # No region detected - will use center crop
        return None

    def _combine_regions(self, regions: List[CropRegion]) -> CropRegion:
        """
        Combine multiple regions into a single bounding region.

        Args:
            regions: List of CropRegion objects

        Returns:
            Single CropRegion encompassing all input regions
        """
        if len(regions) == 1:
            return regions[0]

        min_x = min(r.x for r in regions)
        min_y = min(r.y for r in regions)
        max_x = max(r.x + r.width for r in regions)
        max_y = max(r.y + r.height for r in regions)

        avg_confidence = sum(r.confidence for r in regions) / len(regions)

        return CropRegion(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=avg_confidence
        )

    def _apply_crop(self, image: Image.Image, region: Optional[CropRegion]) -> Image.Image:
        """
        Apply the calculated crop to an image.

        Args:
            image: PIL Image to crop
            region: Region of interest to center on, or None for center crop

        Returns:
            Cropped PIL Image
        """
        crop_box = self._crop_calculator.calculate_crop(image.size, region)
        return image.crop(crop_box)
