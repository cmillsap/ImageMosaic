"""
UI Support Module

Accessibility helpers shared by the main window and the canvas views: text
that keeps its proportions when the system font grows, window sizing that
respects the screen it opens on, text colours that stay readable, and
announcements for screen readers.
"""

import ctypes
import sys

from PyQt6.QtCore import QEvent, QObject, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

# Largest share of the screen's free area the window opens at.
MAX_SCREEN_SHARE = 0.9

# WCAG AA contrast for normal-size text.
MIN_CONTRAST = 4.5

# How far secondary text is faded toward the background before the
# contrast floor is applied.
SECONDARY_FADE = 0.35

# QEvent::ThemeChange, sent when system colours or contrast settings
# change. PyQt6's QEvent.Type doesn't list it.
THEME_CHANGE = QEvent.Type(210)

# Status colours as designed. Each is darkened or lightened as far as the
# theme's background needs for it to stay readable.
STATUS_COLORS = {"done": QColor("#2e8b57"), "problem": QColor("#e74c3c")}


# ============================================================================
# Colour and contrast
# ============================================================================

def relative_luminance(color: QColor) -> float:
    """WCAG relative luminance, 0 for black to 1 for white."""
    def linear(channel):
        c = channel / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return (0.2126 * linear(color.red()) + 0.7152 * linear(color.green())
            + 0.0722 * linear(color.blue()))


def contrast_ratio(a: QColor, b: QColor) -> float:
    """WCAG contrast ratio between two colours, from 1 to 21."""
    darker, lighter = sorted((relative_luminance(a), relative_luminance(b)))
    return (lighter + 0.05) / (darker + 0.05)


def mix(a: QColor, b: QColor, amount: float) -> QColor:
    """a moved amount (0-1) of the way toward b."""
    return QColor(round(a.red() + (b.red() - a.red()) * amount),
                  round(a.green() + (b.green() - a.green()) * amount),
                  round(a.blue() + (b.blue() - a.blue()) * amount))


def readable_color(color: QColor, background: QColor,
                   minimum: float = MIN_CONTRAST) -> QColor:
    """color, darkened or lightened just enough to read on background.

    Keeps the hue where it can, so a brand green stays green, but never
    returns text that fails the contrast floor.
    """
    target = QColor("black") if relative_luminance(background) > 0.18 \
        else QColor("white")
    for step in range(21):
        candidate = mix(color, target, step / 20)
        if contrast_ratio(candidate, background) >= minimum:
            return candidate
    return target


def high_contrast() -> bool:
    """True when the user has asked the system for high contrast.

    The system theme then chooses every colour, and ours stay out of it.
    """
    hints = QGuiApplication.styleHints().accessibility()
    return hints.contrastPreference() == Qt.ContrastPreference.HighContrast


def secondary_text_color(palette: QPalette) -> QColor:
    """Quieter than body text, but still comfortably readable."""
    text = palette.color(QPalette.ColorRole.WindowText)
    background = palette.color(QPalette.ColorRole.Window)
    return readable_color(mix(text, background, SECONDARY_FADE), background)


def tone_color(tone: str, palette: QPalette) -> QColor:
    """The text colour for tone ("secondary", "done", "problem" or None
    for body text) on palette's background.

    Under high contrast every tone is plain body text: the system theme's
    colours are the ones the user can read, and symbols carry the meaning.
    """
    text = palette.color(QPalette.ColorRole.WindowText)
    if tone is None or high_contrast():
        return text
    if tone == "secondary":
        return secondary_text_color(palette)
    return readable_color(STATUS_COLORS[tone],
                          palette.color(QPalette.ColorRole.Window))


class _ToneKeeper(QObject):
    """Re-colours toned labels whenever their colours could have changed:
    a new palette, a system theme or contrast change (which need not come
    with a new palette), or a move to a new parent."""

    EVENTS = (QEvent.Type.PaletteChange, THEME_CHANGE,
              QEvent.Type.ParentChange)

    def eventFilter(self, obj, event):
        if event.type() in self.EVENTS:
            _apply_tone(obj)
        return False


_tone_keeper = None


def set_tone(label: QLabel, tone: str) -> None:
    """Colour label's text by tone, and keep it right as the theme changes.

    tone is "secondary", "done", "problem", or None for body text.
    """
    global _tone_keeper
    if _tone_keeper is None:
        _tone_keeper = _ToneKeeper()
    if label.property("tone_kept") is None:
        label.installEventFilter(_tone_keeper)
        label.setProperty("tone_kept", True)
    label.setProperty("tone", tone)
    _apply_tone(label)


def make_secondary(label: QLabel) -> None:
    """Show label as secondary text: hints, readouts and placeholders.

    Greying text by disabling the label would also tell screen readers
    it is unavailable, so the colour is set directly instead.
    """
    set_tone(label, "secondary")


def _apply_tone(label: QLabel) -> None:
    if label.property("toning"):
        return  # Our own setPalette below echoes back as a PaletteChange.
    parent = label.parentWidget()
    base = parent.palette() if parent else QApplication.palette()
    color = tone_color(label.property("tone"), base)
    palette = QPalette(label.palette())
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        palette.setColor(group, QPalette.ColorRole.WindowText, color)
    label.setProperty("toning", True)
    label.setPalette(palette)
    label.setProperty("toning", False)


# ============================================================================
# Screen reader announcements
# ============================================================================

# Windows UI Automation constants for UiaRaiseNotificationEvent.
_NOTIFICATION_KIND_OTHER = 4
_PROCESSING_IMPORTANT_MOST_RECENT = 1
_PROCESSING_MOST_RECENT = 3

# Replaces the platform announcer, e.g. to record announcements in tests.
_announcer = None


def set_announcer(announcer):
    """Route announce() through announcer(widget, text, important) instead
    of the platform. None restores the platform. Returns the previous one."""
    global _announcer
    previous, _announcer = _announcer, announcer
    return previous


def announce(widget: QWidget, text: str, important: bool = False) -> None:
    """Have a screen reader speak text without moving the keyboard focus.

    For changes the user didn't cause directly, or can't see from where
    they are: a count arriving, a warning appearing, a render finishing.
    important=True lets it interrupt; otherwise it waits its turn and a
    newer message replaces an unspoken older one.
    """
    if not text:
        return
    if _announcer is not None:
        _announcer(widget, text, important)
    else:
        _platform_announce(widget, text, important)


def _platform_announce(widget: QWidget, text: str, important: bool) -> None:
    # PyQt6 doesn't expose Qt's own QAccessibleAnnouncementEvent, so on
    # Windows the notification goes straight to UI Automation, raised on
    # the window. Elsewhere there is no equivalent yet, and the text
    # stays visible on screen for anyone who reads it there.
    if sys.platform != "win32" or QGuiApplication.platformName() != "windows":
        return
    try:
        _uia_notify(int(widget.window().winId()), text, important)
    except (OSError, AttributeError, ValueError):
        pass  # An announcement is never worth crashing the UI over.


def _uia_notify(hwnd: int, text: str, important: bool) -> None:
    uia = ctypes.windll.uiautomationcore
    if not uia.UiaClientsAreListening():
        return  # No screen reader running.

    provider = ctypes.c_void_p()
    uia.UiaHostProviderFromHwnd.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_void_p)]
    if uia.UiaHostProviderFromHwnd(hwnd, ctypes.byref(provider)) or not provider:
        return

    oleaut = ctypes.windll.oleaut32
    oleaut.SysAllocString.restype = ctypes.c_void_p
    oleaut.SysAllocString.argtypes = [ctypes.c_wchar_p]
    oleaut.SysFreeString.argtypes = [ctypes.c_void_p]
    display = oleaut.SysAllocString(text)
    activity = oleaut.SysAllocString("ImageMosaic.Status")
    try:
        uia.UiaRaiseNotificationEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p]
        uia.UiaRaiseNotificationEvent(
            provider, _NOTIFICATION_KIND_OTHER,
            _PROCESSING_IMPORTANT_MOST_RECENT if important
            else _PROCESSING_MOST_RECENT,
            display, activity)
    finally:
        oleaut.SysFreeString(display)
        oleaut.SysFreeString(activity)
        # IUnknown::Release, the third entry in the provider's vtable.
        vtable = ctypes.cast(provider, ctypes.POINTER(
            ctypes.POINTER(ctypes.c_void_p))).contents
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])(provider)


# ============================================================================
# Sizing that follows the system font
# ============================================================================


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
        if not self.wordWrap():
            return
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
        make_secondary(self)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def fits(self) -> bool:
        return self.width() >= self.sizeHint().width()

    def paintEvent(self, event):
        if self.fits():
            super().paintEvent(event)


class StatusLabel(WrapLabel):
    """A short status that reads the same to everyone.

    A done or problem status is marked with a symbol as well as colour,
    and screen readers hear a word in place of the symbol: "Warning: Tile
    is larger than the print" rather than "warning sign Tile is...".
    """

    SYMBOLS = {"done": "✓", "problem": "⚠"}
    SPOKEN = {"done": "Done", "problem": "Warning"}

    def __init__(self, wrap: bool = False, parent=None):
        super().__init__(parent=parent)
        self.setWordWrap(wrap)
        self.kind = None

    def set_status(self, text: str = "", kind: str = None,
                   spoken: str = None) -> None:
        """Show text as kind: None for plain, "done" or "problem".

        spoken replaces what screen readers hear, for a status whose text
        alone would be unclear, such as a bare tick.
        """
        self.kind = kind
        symbol = self.SYMBOLS.get(kind, "")
        self.setText(" ".join(part for part in (symbol, text) if part))
        if spoken is None:
            spoken = f"{self.SPOKEN[kind]}: {text}" if kind and text else text
        self.setAccessibleName(spoken)
        set_tone(self, kind)


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
