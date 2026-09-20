"""
Image Tile Analyzer Module

This module provides functionality to split images into 9 equal sections
and calculate the average color for each section.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
import os


@dataclass
class BoundingBox:
    """Represents a rectangular bounding box."""
    x: int
    y: int
    width: int
    height: int

    def __repr__(self):
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


@dataclass
class ColorAverage:
    """Represents an RGB color average."""
    r: int
    g: int
    b: int

    def as_tuple(self) -> Tuple[int, int, int]:
        """Return color as RGB tuple."""
        return (self.r, self.g, self.b)

    def __repr__(self):
        return f"ColorAverage(r={self.r}, g={self.g}, b={self.b})"


@dataclass
class SectionData:
    """Represents a single section with its bounding box and average color."""
    bounding_box: BoundingBox
    average_color: ColorAverage

    def __repr__(self):
        return f"SectionData(box={self.bounding_box}, color={self.average_color})"


class TileData:
    """
    Data structure for an image tile containing 9 sections.

    Each section has a bounding box and average color.
    Sections are arranged in a 3x3 grid:
        0 1 2
        3 4 5
        6 7 8
    """

    def __init__(self, image_path: str, sections: List[SectionData]):
        """
        Initialize TileData.

        Args:
            image_path: Path to the source image file
            sections: List of 9 SectionData objects
        """
        if len(sections) != 9:
            raise ValueError(f"Expected 9 sections, got {len(sections)}")

        self.image_path = image_path
        self.sections = sections

    def get_section(self, row: int, col: int) -> SectionData:
        """
        Get section data by grid position.

        Args:
            row: Row index (0-2)
            col: Column index (0-2)

        Returns:
            SectionData for the specified position
        """
        if not (0 <= row <= 2 and 0 <= col <= 2):
            raise ValueError(f"Invalid grid position: row={row}, col={col}")

        index = row * 3 + col
        return self.sections[index]

    def get_all_colors(self) -> List[ColorAverage]:
        """Get list of all 9 average colors."""
        return [section.average_color for section in self.sections]

    def get_all_bounding_boxes(self) -> List[BoundingBox]:
        """Get list of all 9 bounding boxes."""
        return [section.bounding_box for section in self.sections]

    def __repr__(self):
        return f"TileData(image_path='{self.image_path}', sections={len(self.sections)})"


class ImageTileAnalyzer:
    """
    Analyzes images by splitting them into a 3x3 grid and calculating
    average colors for each section.
    """

    def analyze_image(self, image_path: str) -> TileData:
        """
        Analyze an image by splitting it into 9 sections.

        Args:
            image_path: Path to the image file

        Returns:
            TileData object containing section information

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be opened or is invalid
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                sections = self._create_sections(img)
                return TileData(image_path, sections)
        except Exception as e:
            raise ValueError(f"Failed to process image {image_path}: {str(e)}")

    def analyze_pil_image(self, img: Image.Image, source_path: str) -> TileData:
        """
        Analyze a PIL Image object directly (for preprocessed images).

        Args:
            img: PIL Image object to analyze
            source_path: Original path to the source image (for reference)

        Returns:
            TileData object containing section information

        Raises:
            ValueError: If image cannot be processed
        """
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            sections = self._create_sections(img)
            return TileData(source_path, sections)
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    def _create_sections(self, img: Image.Image) -> List[SectionData]:
        """
        Split image into 9 sections and calculate average colors.

        Args:
            img: PIL Image object

        Returns:
            List of 9 SectionData objects
        """
        width, height = img.size
        sections = []

        # Calculate section dimensions (as equal as possible)
        section_width = width / 3
        section_height = height / 3

        for row in range(3):
            for col in range(3):
                # Calculate bounding box coordinates
                x1 = int(col * section_width)
                y1 = int(row * section_height)
                x2 = int((col + 1) * section_width)
                y2 = int((row + 1) * section_height)

                # Ensure the last section extends to the edge
                if col == 2:
                    x2 = width
                if row == 2:
                    y2 = height

                # Create bounding box
                bbox = BoundingBox(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1
                )

                # Extract section and calculate average color
                section_img = img.crop((x1, y1, x2, y2))
                avg_color = self._calculate_average_color(section_img)

                sections.append(SectionData(bbox, avg_color))

        return sections

    def _calculate_average_color(self, img: Image.Image) -> ColorAverage:
        """
        Calculate the average RGB color of an image.

        Args:
            img: PIL Image object

        Returns:
            ColorAverage object with average RGB values
        """
        # Averaged in NumPy rather than by summing a Python list of pixel
        # tuples. This runs about ten times faster and it is a hot path:
        # every tile in the library and every cell of the guide passes
        # through here nine times.
        pixels = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
        r, g, b = pixels.mean(axis=0)

        return ColorAverage(r=round(float(r)), g=round(float(g)), b=round(float(b)))
