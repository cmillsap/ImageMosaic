"""
Tests for the canvas views: the finished mosaic panel and the grid preview.
"""

import os

import pytest
from PIL import Image

# Qt needs an offscreen platform plugin under CI / headless runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from result_viewer import GridPreview, MAX_ZOOM, ResultPanel, load_pixmap


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "mosaic.png"
    Image.new('RGB', (2000, 1000), (10, 200, 30)).save(path)
    return str(path)


@pytest.fixture
def viewer(qapp, image_path):
    v = ResultPanel()
    v.resize(900, 900)
    v.show()
    v.show_result(image_path, "9 tiles placed")
    qapp.processEvents()
    yield v
    v.close()


def test_load_pixmap_reads_the_image(qapp, image_path):
    pixmap = load_pixmap(image_path)
    assert (pixmap.width(), pixmap.height()) == (2000, 1000)


def test_load_pixmap_missing_file_is_null(qapp, tmp_path):
    assert load_pixmap(str(tmp_path / "nope.png")).isNull()


def test_opens_fitted_to_the_window(viewer):
    view = viewer.view
    assert view is not None
    assert view.zoom < 1.0
    assert view.zoom == pytest.approx(view.min_zoom(), rel=0.02)


def test_zoom_in_and_back_out_to_fit(viewer):
    view = viewer.view
    fitted = view.zoom
    view.zoom_by(2)
    assert view.zoom == pytest.approx(fitted * 2)
    view.zoom_by(0.01)
    assert view.zoom == pytest.approx(fitted, rel=0.02)


def test_zoom_is_capped(viewer):
    viewer.view.actual_size()
    for _ in range(50):
        viewer.view.zoom_by(2)
    assert viewer.view.zoom == pytest.approx(MAX_ZOOM)


def test_actual_size_is_one_to_one(viewer):
    viewer.view.actual_size()
    assert viewer.view.zoom == 1.0


def test_unreadable_image_shows_a_message(qapp, tmp_path):
    v = ResultPanel()
    v.show_result(str(tmp_path / "missing.png"))
    assert v.view is None
    assert v.fit_btn.isEnabled() is False
    v.close()


def test_summary_is_shown(viewer, image_path):
    text = viewer.info_label.text()
    assert image_path in text and "9 tiles placed" in text


def test_empty_panel_has_nothing_to_act_on(qapp):
    v = ResultPanel()
    assert v.view is None
    assert v.folder_btn.isEnabled() is False
    v.close()


def test_a_second_result_replaces_the_first(viewer, tmp_path, qapp):
    other = tmp_path / "second.png"
    Image.new('RGB', (300, 600), (0, 0, 0)).save(other)
    viewer.show_result(str(other))
    qapp.processEvents()
    rect = viewer.view.sceneRect()
    assert (rect.width(), rect.height()) == (300, 600)


# ============================================================================
# Grid preview
# ============================================================================

@pytest.fixture
def preview(qapp):
    p = GridPreview()
    p.resize(600, 600)
    p.show()
    qapp.processEvents()
    yield p
    p.close()


def test_photo_is_stretched_over_the_whole_grid(preview):
    """GuideImage stretches the guide to the grid, so the preview must too."""
    photo = QPixmap(400, 300)
    preview.set_photo(photo)
    preview.set_grid(60, 90, 100, 100)
    rect = preview.item.sceneBoundingRect()
    assert (rect.width(), rect.height()) == (6000, 9000)


def test_grid_change_rescales_the_photo(preview):
    preview.set_photo(QPixmap(400, 300))
    preview.set_grid(60, 90, 100, 100)
    preview.set_grid(50, 112, 120, 80)
    rect = preview.item.sceneBoundingRect()
    assert (rect.width(), rect.height()) == (6000, 8960)


def test_preview_stays_fitted(preview):
    preview.set_photo(QPixmap(400, 300))
    preview.set_grid(60, 90, 100, 100)
    assert preview.zoom == pytest.approx(preview.min_zoom(), rel=0.02)


def test_empty_grid_shows_the_photo_as_is(preview):
    preview.set_photo(QPixmap(400, 300))
    preview.set_grid(0, 0, 400, 400)
    rect = preview.item.sceneBoundingRect()
    assert (rect.width(), rect.height()) == (400, 300)


def test_grid_can_be_hidden(preview):
    preview.set_show_grid(False)
    assert preview.show_grid is False
