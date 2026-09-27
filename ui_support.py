"""
UI Support Module

Accessibility helpers shared by the main window and the canvas views: text
that keeps its proportions when the system font grows, and window sizing
that respects the screen it opens on.
"""

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

# Largest share of the screen's free area the window opens at.
MAX_SCREEN_SHARE = 0.9


def scale_font(widget: QWidget, factor: float, bold: bool = False) -> None:
    """Size widget's text relative to its window's font, and keep it so.

    Setting a font on a widget pins it: a later change to the system font
    size would leave it behind. Tagged widgets are brought back in line by
    refresh_scaled_fonts, which the window calls on every font change.
    """
    widget.setProperty("font_scale", factor)
    widget.setProperty("font_bold", bold)
    _apply_scaled_font(widget)


def refresh_scaled_fonts(window: QWidget) -> None:
    """Re-derive every scale_font widget in window from its current font."""
    for widget in window.findChildren(QWidget):
        if widget.property("font_scale") is not None:
            _apply_scaled_font(widget)


def _apply_scaled_font(widget: QWidget) -> None:
    window = widget.window()
    font = QApplication.font() if window is widget else window.font()
    font.setPointSizeF(font.pointSizeF() * widget.property("font_scale"))
    font.setBold(bool(widget.property("font_bold")))
    widget.setFont(font)


class WidestTextLabel(QLabel):
    """A label sized for the widest text it will ever show.

    Used for live readouts such as a slider's "off" .. "100%", so the
    layout neither jumps as the value changes nor clips the longest one.
    Measured in the label's own font, so it grows with the system font.
    """

    def __init__(self, widest: str, parent=None):
        super().__init__(parent)
        self.widest = widest

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        margins = self.contentsMargins()
        needed = (self.fontMetrics().horizontalAdvance(self.widest)
                  + margins.left() + margins.right())
        return QSize(max(hint.width(), needed), hint.height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class WrapLabel(QLabel):
    """A word-wrapped label that always has the height its text needs.

    Qt sizes windows without asking wrapped text how tall it becomes at
    the width it is given, so a narrow window could cut the last lines
    off. This label reserves that height itself whenever its width or
    text changes.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)

    def setText(self, text: str) -> None:
        super().setText(text)
        self._reserve_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reserve_height()

    def _reserve_height(self) -> None:
        height = self.heightForWidth(self.width()) if self.text() else 0
        if height != self.minimumHeight():
            self.setMinimumHeight(max(0, height))


class HintLabel(QLabel):
    """A one-line tip that steps aside when there is no room for it.

    Rather than wrap a word per line or push the window wider, the tip
    simply isn't drawn when it doesn't fit, and is shown as a tooltip
    instead. Screen readers still get the full text.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setToolTip(text)
        self.setAlignment(Qt.AlignmentFlag.AlignRight
                          | Qt.AlignmentFlag.AlignVCenter)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def fits(self) -> bool:
        return self.width() >= self.sizeHint().width()

    def paintEvent(self, event):
        if self.fits():
            super().paintEvent(event)


def initial_window_rect(available: QRect, preferred: QSize) -> QRect:
    """Where a window of the preferred size should open on this screen.

    Shrunk to fit when the screen is small, as it is at 150-200% display
    scaling on a laptop, and centred in the space the taskbar leaves free.
    """
    width = min(preferred.width(), int(available.width() * MAX_SCREEN_SHARE))
    height = min(preferred.height(), int(available.height() * MAX_SCREEN_SHARE))
    rect = QRect(0, 0, width, height)
    rect.moveCenter(available.center())
    return rect
