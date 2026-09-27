"""
Tests for the shared accessibility helpers.
"""

import os

import pytest

# Qt needs an offscreen platform plugin under CI / headless runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect, QSize
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from ui_support import (HintLabel, WidestTextLabel, WrapLabel,
                        initial_window_rect, refresh_scaled_fonts, scale_font)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def set_point_size(widget, size):
    font = widget.font()
    font.setPointSize(size)
    widget.setFont(font)


# ============================================================================
# Initial window placement
# ============================================================================

class TestInitialWindowRect:

    def test_large_screen_gets_the_preferred_size_centred(self):
        available = QRect(0, 0, 2560, 1400)
        rect = initial_window_rect(available, QSize(1200, 800))
        assert rect.size() == QSize(1200, 800)
        assert rect.center() == available.center()

    def test_small_screen_shrinks_the_window_to_fit(self):
        # 1920 x 1080 at 200% display scaling, less the taskbar.
        available = QRect(0, 0, 960, 516)
        rect = initial_window_rect(available, QSize(1200, 800))
        assert available.contains(rect)
        assert rect.width() < 960 and rect.height() < 516

    def test_follows_a_screen_that_is_not_at_the_origin(self):
        available = QRect(1920, 0, 1280, 680)
        rect = initial_window_rect(available, QSize(1200, 800))
        assert available.contains(rect)


# ============================================================================
# Relative font sizes
# ============================================================================

class TestScaledFonts:

    @pytest.fixture
    def window(self, qapp):
        window = QWidget()
        window.label = QLabel("Heading")
        QVBoxLayout(window).addWidget(window.label)
        scale_font(window.label, 1.5, bold=True)
        yield window
        window.close()

    def test_scaled_relative_to_the_window(self, window):
        set_point_size(window, 12)
        refresh_scaled_fonts(window)
        assert window.label.font().pointSizeF() == pytest.approx(18)
        assert window.label.font().bold()

    def test_follows_a_later_font_change(self, window):
        for size in (10, 20):
            set_point_size(window, size)
            refresh_scaled_fonts(window)
            assert window.label.font().pointSizeF() == pytest.approx(size * 1.5)


# ============================================================================
# Readouts sized for their widest value
# ============================================================================

class TestWidestTextLabel:

    def test_reserves_room_for_the_widest_text(self, qapp):
        label = WidestTextLabel("100%")
        label.setText("off")
        needed = label.fontMetrics().horizontalAdvance("100%")
        assert label.sizeHint().width() >= needed
        assert label.minimumSizeHint().width() >= needed

    def test_grows_with_the_font(self, qapp):
        label = WidestTextLabel("100%")
        small = label.sizeHint().width()
        set_point_size(label, label.font().pointSize() * 2)
        assert label.sizeHint().width() > small


# ============================================================================
# Wrapped text and optional hints
# ============================================================================

LONG_TEXT = "Choose a photo and a folder of tile photos to start. " * 3


def shown_wrap_label(width):
    """A WrapLabel on screen at width; hidden widgets defer resize events."""
    label = WrapLabel(LONG_TEXT)
    label.show()
    label.resize(width, 10)
    return label


class TestWrapLabel:

    def test_reserves_the_height_its_lines_need(self, qapp):
        label = shown_wrap_label(120)
        assert label.minimumHeight() == label.heightForWidth(120)
        assert label.minimumHeight() > label.fontMetrics().lineSpacing()

    def test_follows_new_text(self, qapp):
        label = shown_wrap_label(120)
        label.setText("Short")
        assert label.minimumHeight() == label.heightForWidth(120)

    def test_empty_label_takes_no_height(self, qapp):
        label = shown_wrap_label(120)
        label.setText("")
        assert label.minimumHeight() == 0


class TestHintLabel:

    def test_can_shrink_to_nothing(self, qapp):
        assert HintLabel("Scroll to zoom").minimumSizeHint().width() == 0

    def test_shown_only_when_it_fits(self, qapp):
        hint = HintLabel("Scroll to zoom, drag to pan")
        hint.resize(hint.sizeHint())
        assert hint.fits()
        hint.resize(hint.sizeHint().width() // 2, hint.height())
        assert not hint.fits()

    def test_text_is_always_available_as_a_tooltip(self, qapp):
        hint = HintLabel("Scroll to zoom, drag to pan")
        assert hint.toolTip() == hint.text()
