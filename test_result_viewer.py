"""
Tests for the ResultViewer window that shows a finished mosaic.
"""

import os

import pytest
from PIL import Image

# Qt needs an offscreen platform plugin under CI / headless runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from result_viewer import MAX_ZOOM, ResultViewer, load_pixmap


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
    v = ResultViewer(image_path, "9 tiles placed")
    v.show()
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
    v = ResultViewer(str(tmp_path / "missing.png"))
    assert v.view is None
    v.close()


def test_close_dismisses(viewer):
    viewer.accept()
    assert not viewer.isVisible()
