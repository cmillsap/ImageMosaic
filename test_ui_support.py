"""
Tests for the shared accessibility helpers.
"""

import os

import pytest

# Qt needs an offscreen platform plugin under CI / headless runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

from PyQt6.QtCore import QRect, QSize
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

import ui_support
from ui_support import (HintLabel, StatusLabel, WidestTextLabel, WrapLabel,
                        announce, contrast_ratio, initial_window_rect,
                        make_secondary, readable_color, refresh_scaled_fonts,
                        scale_font, secondary_text_color, set_announcer)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def themed_palette(text, background):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
    palette.setColor(QPalette.ColorRole.Window, QColor(background))
    return palette


LIGHT = ("#1a1a1a", "#f3f3f3")
DARK = ("#ffffff", "#202020")


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


# ============================================================================
# Contrast
# ============================================================================

class TestContrast:

    def test_known_ratios(self):
        assert contrast_ratio(QColor("black"), QColor("white")) == \
            pytest.approx(21)
        assert contrast_ratio(QColor("#777"), QColor("#777")) == \
            pytest.approx(1)

    def test_is_symmetric(self):
        a, b = QColor("#2e8b57"), QColor("#f3f3f3")
        assert contrast_ratio(a, b) == pytest.approx(contrast_ratio(b, a))

    @pytest.mark.parametrize("color", ["#2e8b57", "#e74c3c", "#999999"])
    @pytest.mark.parametrize("background", ["#ffffff", "#f3f3f3", "#202020"])
    def test_readable_color_meets_the_floor(self, color, background):
        fixed = readable_color(QColor(color), QColor(background))
        assert contrast_ratio(fixed, QColor(background)) >= 4.5

    def test_readable_color_leaves_good_colours_alone(self):
        assert readable_color(QColor("black"), QColor("white")) == \
            QColor("black")

    @pytest.mark.parametrize("scheme", [LIGHT, DARK])
    def test_secondary_text_is_quieter_but_readable(self, scheme):
        palette = themed_palette(*scheme)
        color = secondary_text_color(palette)
        background = QColor(scheme[1])
        assert contrast_ratio(color, background) >= 4.5
        assert contrast_ratio(color, background) < \
            contrast_ratio(QColor(scheme[0]), background)


# ============================================================================
# Secondary text
# ============================================================================

class TestSecondaryText:

    def text_color(self, label):
        return label.palette().color(QPalette.ColorGroup.Active,
                                     QPalette.ColorRole.WindowText)

    def test_stays_enabled(self, qapp):
        label = QLabel("Hint")
        make_secondary(label)
        assert label.isEnabled()

    def test_follows_a_theme_change(self, qapp):
        parent = QWidget()
        parent.setPalette(themed_palette(*LIGHT))
        label = QLabel("Hint", parent)
        make_secondary(label)
        light = self.text_color(label)

        parent.setPalette(themed_palette(*DARK))
        dark = self.text_color(label)
        assert dark != light
        assert contrast_ratio(dark, QColor(DARK[1])) >= 4.5

    def test_follows_a_new_parent(self, qapp):
        label = QLabel("Hint")
        make_secondary(label)
        parent = QWidget()
        parent.setPalette(themed_palette(*DARK))
        label.setParent(parent)
        assert contrast_ratio(self.text_color(label), QColor(DARK[1])) >= 4.5

    def test_high_contrast_uses_plain_text(self, qapp, monkeypatch):
        monkeypatch.setattr(ui_support, "high_contrast", lambda: True)
        parent = QWidget()
        parent.setPalette(themed_palette(*LIGHT))
        label = QLabel("Hint", parent)
        make_secondary(label)
        assert self.text_color(label) == QColor(LIGHT[0])

    def test_hints_are_secondary(self, qapp):
        assert HintLabel("Scroll").property("tone") == "secondary"


# ============================================================================
# Status labels
# ============================================================================

class TestStatusLabel:

    def test_problem_has_a_symbol_and_a_spoken_word(self, qapp):
        label = StatusLabel()
        label.set_status("Folder not found", "problem")
        assert label.text() == "⚠ Folder not found"
        assert label.accessibleName() == "Warning: Folder not found"

    def test_done_has_a_tick(self, qapp):
        label = StatusLabel()
        label.set_status("3 found", "done")
        assert label.text() == "✓ 3 found"
        assert label.accessibleName() == "Done: 3 found"

    def test_spoken_text_can_be_given(self, qapp):
        label = StatusLabel()
        label.set_status(kind="done", spoken="Photo chosen")
        assert label.text() == "✓"
        assert label.accessibleName() == "Photo chosen"

    def test_plain_status_has_no_symbol(self, qapp):
        label = StatusLabel()
        label.set_status("60 × 90 grid")
        assert label.text() == "60 × 90 grid"
        assert label.kind is None

    def test_clearing(self, qapp):
        label = StatusLabel()
        label.set_status("Folder not found", "problem")
        label.set_status()
        assert label.text() == "" and label.accessibleName() == ""

    def test_wraps_only_when_asked(self, qapp):
        assert not StatusLabel().wordWrap()
        assert StatusLabel(wrap=True).wordWrap()


# ============================================================================
# Announcements
# ============================================================================

class TestAnnounce:

    @pytest.fixture
    def heard(self):
        heard = []
        previous = set_announcer(lambda w, text, important:
                                 heard.append((text, important)))
        yield heard
        set_announcer(previous)

    def test_routes_to_the_announcer(self, qapp, heard):
        announce(QWidget(), "Mosaic saved", important=True)
        assert heard == [("Mosaic saved", True)]

    def test_empty_text_is_not_announced(self, qapp, heard):
        announce(QWidget(), "")
        assert heard == []

    def test_set_announcer_returns_the_previous_one(self, qapp):
        first = lambda *a: None
        previous = set_announcer(first)
        try:
            assert set_announcer(previous) is first
        finally:
            set_announcer(previous)

    def test_platform_announcer_never_raises(self, qapp):
        # Offscreen there is no native window; it must simply do nothing.
        announce(QWidget(), "Mosaic saved")

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows UI Automation")
    def test_uia_notify_tolerates_a_bad_window(self, qapp):
        ui_support._uia_notify(0, "Mosaic saved", False)
