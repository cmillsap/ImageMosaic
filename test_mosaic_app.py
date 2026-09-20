"""
Tests for the MosaicApp UI's derived sizing logic.

These cover the pure calculations behind the tile grid - the numbers that
GuideImage and TilePreprocessor will both be driven from - rather than
widget appearance.
"""

import os

import pytest

# Qt needs an offscreen platform plugin under CI / headless runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from mosaic_app import MosaicApp
from tile_preprocessor import PreprocessorConfig


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for the whole test session."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    """A fresh MosaicApp window per test."""
    w = MosaicApp()
    yield w
    w.close()


def configure(window, out_w_in, out_h_in, tile_w, tile_h):
    """Drive the window through its spin boxes, as a user would."""
    window.width_spinbox.setValue(out_w_in)
    window.height_spinbox.setValue(out_h_in)
    window.tile_width_spinbox.setValue(tile_w)
    window.tile_height_spinbox.setValue(tile_h)


# ============================================================================
# Output pixel dimensions
# ============================================================================

class TestOutputDimensions:

    def test_defaults(self, window):
        assert window.output_dpi == 300
        assert window.output_width_px == 20 * 300
        assert window.output_height_px == 30 * 300

    def test_tracks_spinboxes(self, window):
        configure(window, 10, 8, 100, 100)
        assert window.output_width_px == 3000
        assert window.output_height_px == 2400

    def test_label_reports_pixels_and_dpi(self, window):
        configure(window, 10, 8, 100, 100)
        text = window.output_dimensions_label.text()
        assert "3000" in text and "2400" in text and "300 DPI" in text


# ============================================================================
# Tile grid math
# ============================================================================

class TestTileGrid:

    def test_square_tiles_divide_evenly(self, window):
        configure(window, 20, 30, 100, 100)
        assert (window.grid_cols, window.grid_rows) == (60, 90)
        assert window.total_tiles == 5400
        assert window.remainder_px == (0, 0)

    def test_non_square_tiles(self, window):
        configure(window, 20, 30, 120, 80)
        # 6000 // 120 = 50, 9000 // 80 = 112 remainder 40
        assert (window.grid_cols, window.grid_rows) == (50, 112)
        assert window.total_tiles == 5600
        assert window.remainder_px == (0, 40)

    def test_remainder_is_reported_in_label(self, window):
        configure(window, 20, 30, 120, 80)
        assert "unused" in window.grid_info_label.text()

    def test_no_remainder_omits_unused_note(self, window):
        configure(window, 20, 30, 100, 100)
        assert "unused" not in window.grid_info_label.text()

    def test_grid_uses_floor_division(self, window):
        # 1 in = 300 px; a 7 px tile leaves 300 % 7 = 6 px over per axis.
        configure(window, 1, 1, 8, 8)
        assert window.grid_cols == 300 // 8
        assert window.remainder_px == (300 - 37 * 8, 300 - 37 * 8)

    def test_aspect_ratio(self, window):
        configure(window, 20, 30, 120, 80)
        assert window.tile_aspect_ratio == pytest.approx(1.5)
        assert "1.500" in window.tile_aspect_label.text()


# ============================================================================
# Degenerate configuration
# ============================================================================

class TestOversizedTile:

    def test_tile_larger_than_canvas_yields_empty_grid(self, window):
        configure(window, 1, 1, 400, 400)  # 300x300 px canvas
        assert window.grid_cols == 0
        assert window.grid_rows == 0
        assert window.total_tiles == 0

    def test_warning_is_shown(self, window):
        configure(window, 1, 1, 400, 400)
        assert "no tiles fit" in window.grid_info_label.text()

    def test_warning_clears_when_tile_shrinks(self, window):
        configure(window, 1, 1, 400, 400)
        assert "no tiles fit" in window.grid_info_label.text()

        window.tile_width_spinbox.setValue(100)
        window.tile_height_spinbox.setValue(100)
        assert "no tiles fit" not in window.grid_info_label.text()
        assert window.total_tiles == 9


# ============================================================================
# Preprocessor config derivation
# ============================================================================

class TestPreprocessorConfigDerivation:
    """The UI is the single source of truth for tile dimensions.

    GuideImage cuts cells at these dimensions and TilePreprocessor crops tiles
    to this aspect ratio; if the two disagreed, every tile would be distorted.
    """

    def test_returns_a_config(self, window):
        assert isinstance(window.build_preprocessor_config(), PreprocessorConfig)

    def test_config_matches_tile_spinboxes(self, window):
        configure(window, 20, 30, 120, 80)
        cfg = window.build_preprocessor_config()
        assert cfg.target_width == 120
        assert cfg.target_height == 80

    def test_config_aspect_ratio_matches_grid_cell(self, window):
        configure(window, 20, 30, 120, 80)
        cfg = window.build_preprocessor_config()
        assert cfg.target_aspect_ratio == pytest.approx(window.tile_aspect_ratio)

    def test_config_follows_later_edits(self, window):
        configure(window, 20, 30, 100, 100)
        assert window.build_preprocessor_config().target_width == 100

        window.tile_width_spinbox.setValue(250)
        assert window.build_preprocessor_config().target_width == 250

    def test_overrides_are_applied(self, window):
        cfg = window.build_preprocessor_config(
            cache_dir="/tmp/tiles", enable_saliency=False
        )
        assert cfg.cache_dir == "/tmp/tiles"
        assert cfg.enable_saliency is False
        # Tile dimensions still come from the UI.
        assert cfg.target_width == window.tile_width

    def test_override_can_replace_tile_dimensions(self, window):
        cfg = window.build_preprocessor_config(target_width=64)
        assert cfg.target_width == 64


# ============================================================================
# Existing behaviour that the new controls must not break
# ============================================================================

class TestExistingControls:

    def test_subdirectory_checkbox_round_trip(self, window):
        assert window.scan_subdirectories is False
        window.subdirs_checkbox.setChecked(True)
        assert window.scan_subdirectories is True
        window.subdirs_checkbox.setChecked(False)
        assert window.scan_subdirectories is False

    def test_generate_disabled_until_both_inputs_chosen(self, window):
        assert window.generate_btn.isEnabled() is False

        window.guide_image_path = "guide.png"
        window.check_ready_to_generate()
        assert window.generate_btn.isEnabled() is False

        window.tile_folder_path = "tiles/"
        window.check_ready_to_generate()
        assert window.generate_btn.isEnabled() is True

    def test_progress_starts_hidden(self, window):
        assert window.progress_group.isVisible() is False
