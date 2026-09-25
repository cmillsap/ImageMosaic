"""
Tests for the MosaicApp UI: the derived sizing calculations behind the tile
grid, and the wiring that runs a real render on a worker thread.
"""

import os

import pytest
from PIL import Image

# Qt needs an offscreen platform plugin under CI / headless runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QThread, QStandardPaths, QSettings
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from mosaic_app import LOOK_PRESETS, PRINT_SIZES, MosaicApp
from result_viewer import ResultPanel
from tile_preprocessor import PreprocessorConfig


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for the whole test session."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def settings(tmp_path):
    """Settings in a throwaway file, never the user's real preferences."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(qapp, settings):
    """A fresh MosaicApp window per test."""
    w = MosaicApp(settings)
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
        assert window.progress_widget.isHidden() is True


# ============================================================================
# The demo animation must be gone, not merely unused
# ============================================================================

def _app_source():
    import mosaic_app
    path = os.path.join(os.path.dirname(mosaic_app.__file__), 'mosaic_app.py')
    with open(path, encoding='utf-8') as handle:
        return handle.read()


class TestDemoCodeRemoved:

    @pytest.mark.parametrize("name", [
        "demo_generate_mosaic", "_start_next_demo_task", "_demo_tick",
    ])
    def test_demo_methods_are_gone(self, window, name):
        assert not hasattr(window, name)

    def test_no_demo_code_remains(self):
        assert 'demo' not in _app_source().lower()

    def test_gui_thread_never_pumps_the_event_loop(self):
        """processEvents() was how the demo faked responsiveness while
        blocking the GUI thread. Real work runs on a worker now."""
        assert 'processEvents' not in _app_source()


# ============================================================================
# Job snapshot
# ============================================================================

class TestJobSnapshot:

    def _job(self, window, tmp):
        window.guide_image_path = os.path.join(tmp, 'guide.png')
        window.tile_folder_path = os.path.join(tmp, 'tiles')
        configure(window, 20, 30, 120, 80)
        return window.build_job(os.path.join(tmp, 'out.png'))

    def test_captures_paths_and_dimensions(self, window, tmp_path):
        job = self._job(window, str(tmp_path))
        assert job.guide_image_path == window.guide_image_path
        assert job.tile_folder_path == window.tile_folder_path
        assert job.tile_width == 120
        assert job.tile_height == 80
        assert job.output_width_px == window.output_width_px
        assert job.output_height_px == window.output_height_px
        assert job.output_dpi == window.output_dpi

    def test_captures_variety_settings(self, window, tmp_path):
        window.max_reuse_spinbox.setValue(6)
        window.min_distance_spinbox.setValue(3)
        job = self._job(window, str(tmp_path))
        assert job.max_tile_reuse == 6
        assert job.min_reuse_distance == 3

    def test_captures_variety_tint_and_order(self, window, tmp_path):
        window.variety_slider.setValue(35)
        window.tint_slider.setValue(20)
        window.randomize_order_checkbox.setChecked(False)
        job = self._job(window, str(tmp_path))
        assert job.variety == 35
        assert job.tint_strength == 20
        assert job.randomize_order is False

    def test_variety_and_tint_start_off(self, window, tmp_path):
        assert window.variety_value_label.text() == "off"
        assert window.tint_value_label.text() == "off"
        job = self._job(window, str(tmp_path))
        assert job.variety == 0
        assert job.tint_strength == 0
        assert job.randomize_order is True

    def test_variety_and_tint_are_bounded(self, window):
        for box in (window.variety_slider, window.tint_slider):
            assert (box.minimum(), box.maximum()) == (0, 100)

    def test_captures_subdirectory_flag(self, window, tmp_path):
        window.subdirs_checkbox.setChecked(True)
        job = self._job(window, str(tmp_path))
        assert job.scan_subdirectories is True

    def test_is_a_snapshot_not_a_live_view(self, window, tmp_path):
        """The worker must not read a control the user is still editing."""
        job = self._job(window, str(tmp_path))
        window.tile_width_spinbox.setValue(40)
        assert job.tile_width == 120

    def test_cache_dir_is_set(self, window, tmp_path):
        """Tiles get prepared twice - indexed, then pasted - so the
        on-disk cache is what stops the second pass repeating the work."""
        assert self._job(window, str(tmp_path)).cache_dir


# ============================================================================
# Guards before a run starts
# ============================================================================

class TestGenerateGuards:

    def test_refuses_when_no_tiles_fit(self, window, monkeypatch):
        configure(window, 1, 1, 400, 400)   # 300x300 canvas, 400px tile
        assert window.total_tiles == 0

        warned = []
        monkeypatch.setattr(QMessageBox, 'warning',
                            lambda *a, **k: warned.append(a))
        monkeypatch.setattr(window, 'choose_output_path',
                            lambda: pytest.fail("reached the save dialog"))

        window.generate_mosaic()
        assert warned
        assert window.worker_thread is None

    def test_cancelling_the_save_dialog_starts_nothing(self, window, monkeypatch):
        configure(window, 20, 30, 100, 100)
        monkeypatch.setattr(window, 'choose_output_path', lambda: None)

        window.generate_mosaic()
        assert window.worker is None
        assert window.worker_thread is None

    def test_will_not_start_a_second_run(self, window, monkeypatch):
        window.worker_thread = object()          # pretend one is running
        monkeypatch.setattr(window, 'choose_output_path',
                            lambda: pytest.fail("started a second run"))
        window.generate_mosaic()
        window.worker_thread = None              # tidy up for teardown


# ============================================================================
# Output location
# ============================================================================

class TestOutputLocation:

    def test_defaults_to_documents(self, window):
        documents = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation)
        assert window.output_folder == documents
        assert window.output_folder_edit.text() == documents

    def test_browse_sets_the_folder(self, window, monkeypatch, tmp_path):
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *a, **k: str(tmp_path))
        window.select_output_folder()
        assert window.output_folder == str(tmp_path)
        assert window.output_folder_edit.text() == str(tmp_path)

    def test_browsed_folder_is_remembered(self, window, qapp, settings,
                                          monkeypatch, tmp_path):
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *a, **k: str(tmp_path))
        window.select_output_folder()

        reopened = MosaicApp(settings)
        assert reopened.output_folder == str(tmp_path)
        assert reopened.output_folder_edit.text() == str(tmp_path)
        reopened.close()

    def test_missing_remembered_folder_falls_back(self, qapp, settings, tmp_path):
        settings.setValue("output_folder", str(tmp_path / "deleted"))
        w = MosaicApp(settings)
        assert w.output_folder == QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation)
        w.close()

    def test_cancelled_browse_keeps_the_folder(self, window, monkeypatch):
        before = window.output_folder
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *a, **k: "")
        window.select_output_folder()
        assert window.output_folder == before

    def test_path_is_named_after_the_guide(self, window, tmp_path):
        window.output_folder = str(tmp_path)
        window.guide_image_path = os.path.join("somewhere", "sunset.jpg")
        assert window.choose_output_path() == str(tmp_path / "sunset_mosaic.png")

    def test_path_follows_the_chosen_format(self, window, tmp_path):
        window.output_folder = str(tmp_path)
        window.output_format_combo.setCurrentIndex(
            window.output_format_combo.findData(".jpg"))
        assert window.choose_output_path() == str(tmp_path / "mosaic.jpg")

    def test_existing_mosaic_is_not_overwritten(self, window, tmp_path):
        window.output_folder = str(tmp_path)
        (tmp_path / "mosaic.png").touch()
        (tmp_path / "mosaic (2).png").touch()
        assert window.choose_output_path() == str(tmp_path / "mosaic (3).png")


# ============================================================================
# Progress display
# ============================================================================

class TestProgressUi:

    def test_show_progress_reveals_cancel(self, window):
        window.show_progress()
        assert window.cancel_btn.isHidden() is False
        assert window.generate_btn.isEnabled() is False

    def test_hide_progress_restores_idle_state(self, window):
        window.show_progress()
        window.hide_progress()
        assert window.progress_widget.isHidden() is True
        assert window.cancel_btn.isHidden() is True
        assert window.progress_status_label.text() == "Ready"

    def test_stage_progress_shows_counts(self, window):
        window.on_stage_progress("Loading tiles...", 5, 1000)
        text = window.progress_status_label.text()
        assert "Loading tiles..." in text
        assert "1,000" in text

    def test_single_step_stage_omits_counts(self, window):
        window.on_stage_progress("Saving...", 1, 1)
        assert window.progress_status_label.text() == "Saving..."

    def test_progress_value_is_clamped(self, window):
        window.set_progress_value(500)
        assert window.progress_bar.value() == 100
        window.set_progress_value(-20)
        assert window.progress_bar.value() == 0


# ============================================================================
# Worker lifecycle, driving a real thread
# ============================================================================

class TestWorkerLifecycle:

    def _start(self, window, monkeypatch, tmp):
        tiles = os.path.join(tmp, 'tiles')
        os.makedirs(tiles, exist_ok=True)
        for i, color in enumerate([(255, 0, 0), (0, 0, 255), (0, 255, 0)]):
            Image.new('RGB', (20, 20), color).save(
                os.path.join(tiles, f't{i}.png'))

        guide = os.path.join(tmp, 'guide.png')
        Image.new('RGB', (40, 40), (255, 0, 0)).save(guide)

        window.guide_image_path = guide
        window.tile_folder_path = tiles
        configure(window, 1, 1, 100, 100)     # 300x300 canvas -> 3x3 grid

        output = os.path.join(tmp, 'out.png')
        monkeypatch.setattr(window, 'choose_output_path', lambda: output)
        monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: None)
        monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: None)

        window.generate_mosaic()
        return output

    def _drain(self, window, qapp):
        while window.worker_thread is not None:
            qapp.processEvents()

    def test_render_runs_off_the_gui_thread(self, window, monkeypatch, qapp, tmp_path):
        output = self._start(window, monkeypatch, str(tmp_path))

        assert isinstance(window.worker_thread, QThread)
        assert window.worker.thread() is window.worker_thread
        assert window.worker.thread() is not QThread.currentThread()

        self._drain(window, qapp)
        assert os.path.exists(output)

    def test_teardown_clears_both_objects(self, window, monkeypatch, qapp, tmp_path):
        self._start(window, monkeypatch, str(tmp_path))
        self._drain(window, qapp)

        assert window.worker is None
        assert window.worker_thread is None

    def test_ui_returns_to_idle_after_success(self, window, monkeypatch, qapp, tmp_path):
        self._start(window, monkeypatch, str(tmp_path))
        self._drain(window, qapp)

        assert window.progress_widget.isHidden() is True
        assert window.generate_btn.isEnabled() is True

    def test_result_opens_in_the_mosaic_tab(self, window, monkeypatch, qapp, tmp_path):
        output = self._start(window, monkeypatch, str(tmp_path))
        self._drain(window, qapp)

        assert isinstance(window.result_panel, ResultPanel)
        assert window.canvas_tabs.isTabEnabled(1)
        assert window.canvas_tabs.currentWidget() is window.result_panel
        assert window.result_panel.image_path == output
        assert window.result_panel.view is not None

    def test_cancel_button_stops_the_run(self, window, monkeypatch, qapp, tmp_path):
        self._start(window, monkeypatch, str(tmp_path))
        window.cancel_btn.click()
        self._drain(window, qapp)

        assert window.worker is None
        assert window.generate_btn.isEnabled() is True

    def test_closing_the_window_stops_the_worker(self, window, monkeypatch, qapp, tmp_path):
        self._start(window, monkeypatch, str(tmp_path))
        window.close()

        assert window.worker is None
        assert window.worker_thread is None


# ============================================================================
# Print size presets
# ============================================================================

class TestPrintSize:

    def test_default_is_a_preset_with_fields_hidden(self, window):
        assert window.print_size_combo.currentData() == (20, 30)
        assert window.custom_print_row.isHidden() is True

    def test_choosing_a_preset_sets_the_inches(self, window):
        window.print_size_combo.setCurrentIndex(
            PRINT_SIZES.index((16, 20)))
        assert (window.output_width_inches, window.output_height_inches) == (16, 20)
        assert window.total_tiles == (16 * 300 // 100) * (20 * 300 // 100)

    def test_preset_keeps_landscape_orientation(self, window):
        window.swap_orientation()
        window.print_size_combo.setCurrentIndex(
            PRINT_SIZES.index((8, 10)))
        assert (window.output_width_inches, window.output_height_inches) == (10, 8)

    def test_swap_orientation(self, window):
        window.swap_orientation()
        assert (window.output_width_inches, window.output_height_inches) == (30, 20)
        assert window.print_size_combo.currentData() == (20, 30)
        assert "landscape" in window.output_dimensions_label.text()

    def test_unmatched_size_switches_to_custom(self, window):
        configure(window, 10, 8, 100, 100)
        assert window.print_size_combo.currentData() is None
        assert window.custom_print_row.isHidden() is False

    def test_custom_stays_open_while_typing_a_preset_size(self, window):
        window.print_size_combo.setCurrentIndex(
            window.print_size_combo.count() - 1)
        window.width_spinbox.setValue(16)
        window.height_spinbox.setValue(20)
        assert window.print_size_combo.currentData() is None
        assert window.custom_print_row.isHidden() is False


# ============================================================================
# Tile shape buttons
# ============================================================================

class TestTileShape:

    def test_square_by_default(self, window):
        assert window.tile_shape_group.checkedId() == 0
        assert window.custom_tile_row.isHidden() is True

    def test_width_drives_height_for_a_fixed_shape(self, window):
        window.tile_width_spinbox.setValue(80)
        assert window.tile_height == 80

    def test_choosing_a_shape_derives_height(self, window):
        window.tile_shape_buttons[1].click()     # 4:3
        assert (window.tile_width, window.tile_height) == (100, 75)
        window.tile_width_spinbox.setValue(120)
        assert window.tile_height == 90

    def test_setting_height_selects_the_matching_shape(self, window):
        configure(window, 20, 30, 120, 80)
        assert window.tile_shape_group.checkedId() == 2    # 3:2
        assert window.custom_tile_row.isHidden() is True

    def test_odd_height_switches_to_custom(self, window):
        window.tile_height_spinbox.setValue(37)
        assert window.tile_shape_buttons[-1].isChecked()
        assert window.custom_tile_row.isHidden() is False
        window.tile_width_spinbox.setValue(50)
        assert window.tile_height == 37             # custom: width is free

    def test_custom_button_reveals_height(self, window):
        window.tile_shape_buttons[-1].click()
        assert window.custom_tile_row.isHidden() is False
        assert window.tile_size_label.text() == "Width"


# ============================================================================
# Look presets
# ============================================================================

class TestLookPresets:

    def test_starts_on_accurate(self, window):
        assert window.current_look_preset() == "Accurate"
        assert window.look_buttons["Accurate"].isChecked()

    @pytest.mark.parametrize("name", list(LOOK_PRESETS))
    def test_preset_sets_every_control(self, window, tmp_path, name):
        window.randomize_order_checkbox.setChecked(False)
        window.look_buttons[name].click()
        values = LOOK_PRESETS[name]
        window.guide_image_path = "guide.png"
        window.tile_folder_path = "tiles"
        job = window.build_job(str(tmp_path / "out.png"))
        assert job.variety == values["variety"]
        assert job.tint_strength == values["tint"]
        assert job.min_reuse_distance == values["min_gap"]
        assert job.max_tile_reuse == values["max_uses"]
        assert job.randomize_order is True
        assert window.look_buttons[name].isChecked()

    def test_nudging_a_control_shows_custom(self, window):
        window.look_buttons["Balanced"].click()
        window.variety_slider.setValue(31)
        assert window.current_look_preset() is None
        assert not any(b.isChecked() for b in window.look_buttons.values())
        assert window.look_custom_label.text() == "Custom"

    def test_returning_to_preset_values_reselects_it(self, window):
        window.variety_slider.setValue(30)
        window.tint_slider.setValue(15)
        window.min_distance_spinbox.setValue(2)
        assert window.look_buttons["Balanced"].isChecked()
        assert window.look_custom_label.text() == ""

    def test_advanced_starts_collapsed(self, window):
        assert window.advanced_panel.isHidden() is True
        window.advanced_toggle.setChecked(True)
        assert window.advanced_panel.isHidden() is False


# ============================================================================
# Photo, tile folder and the reason Generate is disabled
# ============================================================================

class TestInputs:

    def test_hint_names_what_is_missing(self, window):
        assert "photo and a folder" in window.generate_hint_label.text()
        window.guide_image_path = "guide.png"
        window.check_ready_to_generate()
        assert "folder" in window.generate_hint_label.text()
        window.tile_folder_path = "tiles"
        window.check_ready_to_generate()
        assert window.generate_hint_label.isHidden() is True

    def test_no_tiles_fit_disables_generate(self, window):
        window.guide_image_path = "guide.png"
        window.tile_folder_path = "tiles"
        configure(window, 1, 1, 400, 400)
        assert window.generate_btn.isEnabled() is False
        assert "smaller" in window.generate_hint_label.text()

    def test_tile_folder_is_counted(self, window, tmp_path):
        for name in ("a.jpg", "b.PNG", "notes.txt"):
            (tmp_path / name).touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.jpg").touch()

        window.set_tile_folder(str(tmp_path))
        assert window.tile_count == 2
        assert "2 found" in window.tiles_status_label.text()

        window.subdirs_checkbox.setChecked(True)
        assert window.tile_count == 3

    def test_empty_tile_folder_blocks_generate(self, window, tmp_path):
        window.guide_image_path = "guide.png"
        window.set_tile_folder(str(tmp_path))
        assert window.tile_count == 0
        assert window.generate_btn.isEnabled() is False
        assert "No images" in window.generate_hint_label.text()

    def test_missing_tile_folder_is_reported(self, window, tmp_path):
        window.set_tile_folder(str(tmp_path / "gone"))
        assert window.tile_count is None
        assert "not found" in window.tiles_status_label.text()

    def test_set_guide_image_shows_it(self, window, tmp_path):
        path = tmp_path / "sunset.jpg"
        Image.new('RGB', (400, 300), (250, 120, 60)).save(path)
        assert window.set_guide_image(str(path)) is True
        assert window.guide_image_path == str(path)
        assert "400 × 300" in window.guide_size_label.text()
        assert window.preview_stack.currentWidget() is window.grid_preview
        assert window.output_name_label.text() == "sunset_mosaic.png"

    def test_unreadable_guide_is_rejected(self, window, tmp_path, monkeypatch):
        path = tmp_path / "broken.jpg"
        path.write_bytes(b"not an image")
        warned = []
        monkeypatch.setattr(QMessageBox, 'warning',
                            lambda *a, **k: warned.append(a))
        assert window.set_guide_image(str(path)) is False
        assert warned
        assert window.guide_image_path is None

    def test_output_name_follows_format(self, window):
        window.output_format_combo.setCurrentIndex(
            window.output_format_combo.findData(".tif"))
        assert window.output_name_label.text() == "mosaic.tif"

    def test_grid_preview_tracks_the_size_controls(self, window):
        configure(window, 20, 30, 120, 80)
        assert (window.grid_preview.cols, window.grid_preview.rows) == (50, 112)


class TestStretchWarning:

    def _guide(self, window, tmp_path, size):
        path = tmp_path / "guide.png"
        Image.new('RGB', size, (0, 0, 0)).save(path)
        window.set_guide_image(str(path))

    def test_matching_shape_has_no_warning(self, window, tmp_path):
        self._guide(window, tmp_path, (200, 300))       # 2:3, like 20x30
        assert window.photo_stretch == pytest.approx(1.0)
        assert window.stretch_label.isHidden() is True

    def test_wide_photo_on_tall_print_warns(self, window, tmp_path):
        self._guide(window, tmp_path, (300, 200))
        assert window.photo_stretch < 1
        assert window.stretch_label.isHidden() is False
        assert "taller by 125%" in window.stretch_label.text()

    def test_swapping_orientation_clears_it(self, window, tmp_path):
        self._guide(window, tmp_path, (300, 200))
        window.swap_orientation()
        assert window.stretch_label.isHidden() is True

    def test_no_photo_no_warning(self, window):
        assert window.photo_stretch is None
        assert window.stretch_label.isHidden() is True

    def test_preview_controls_wait_for_a_photo(self, window, tmp_path):
        assert window.preview_controls.isEnabled() is False
        self._guide(window, tmp_path, (200, 300))
        assert window.preview_controls.isEnabled() is True
