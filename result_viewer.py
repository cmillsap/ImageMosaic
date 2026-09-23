"""
Result Viewer Module

A window that shows the finished mosaic so it can be inspected up close.

A mosaic is only interesting at two scales: from across the room, where the
guide image appears, and nose-to-the-glass, where the individual photos do.
The viewer opens fitted to the window and zooms with the mouse wheel down to
actual pixels, panning by drag.
"""

import os

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QImageReader, QPainter, QPixmap
from PyQt6.QtWidgets import (QDialog, QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout)

# Wheel step and the zoom range it is clamped to, as a scale factor where
# 1.0 is one image pixel per screen pixel.
ZOOM_STEP = 1.25
MAX_ZOOM = 8.0


def load_pixmap(path: str) -> QPixmap:
    """Load an image of any size, returning a null pixmap on failure.

    Qt refuses to decode images over 256 MB by default, and a large print
    mosaic (e.g. 30000 x 30000 px) is well past that. The file is our own
    output, so the limit is lifted for this one read.
    """
    reader = QImageReader(path)
    reader.setAllocationLimit(0)
    image = reader.read()
    if image.isNull():
        return QPixmap()
    return QPixmap.fromImage(image)


class ZoomableImageView(QGraphicsView):
    """A QGraphicsView that zooms under the cursor and pans by drag."""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem(pixmap)
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        # Stays in "fit" mode, refitting on resize, until the user zooms.
        self._fitted = True

    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def min_zoom(self) -> float:
        """The zoom at which the whole image fits; never zoom out past it."""
        rect = self.item.boundingRect()
        if rect.isEmpty():
            return 1.0
        viewport = self.viewport().rect()
        return min(viewport.width() / rect.width(),
                   viewport.height() / rect.height(), 1.0)

    def fit(self) -> None:
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
        self._fitted = True

    def actual_size(self) -> None:
        self.resetTransform()
        self._fitted = False

    def zoom_by(self, factor: float) -> None:
        target = max(self.min_zoom(), min(MAX_ZOOM, self.zoom * factor))
        if target <= self.min_zoom():
            self.fit()
            return
        self.scale(target / self.zoom, target / self.zoom)
        self._fitted = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(ZOOM_STEP if delta > 0 else 1 / ZOOM_STEP)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fitted:
            self.fit()


class ResultViewer(QDialog):
    """Shows a finished mosaic with its summary until the user closes it."""

    def __init__(self, image_path: str, summary: str = "", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setWindowTitle(f"Mosaic - {os.path.basename(image_path)}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(900, 900)

        layout = QVBoxLayout(self)

        pixmap = load_pixmap(image_path)
        if pixmap.isNull():
            self.view = None
            message = QLabel(f"The mosaic was saved, but could not be "
                             f"displayed:\n{image_path}")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(message, 1)
        else:
            self.view = ZoomableImageView(pixmap, self)
            layout.addWidget(self.view, 1)

        info = QLabel(f"Saved to: {image_path}"
                      + (f"\n{summary}" if summary else ""))
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setWordWrap(True)
        layout.addWidget(info)

        buttons = QHBoxLayout()
        if self.view is not None:
            fit_btn = QPushButton("Fit to Window")
            fit_btn.clicked.connect(self.view.fit)
            buttons.addWidget(fit_btn)

            actual_btn = QPushButton("Actual Size")
            actual_btn.clicked.connect(self.view.actual_size)
            buttons.addWidget(actual_btn)

            hint = QLabel("Scroll to zoom, drag to pan")
            hint.setEnabled(False)  # palette-aware secondary text
            buttons.addWidget(hint)

        buttons.addStretch()

        folder_btn = QPushButton("Open Folder")
        folder_btn.clicked.connect(self.open_folder)
        buttons.addWidget(folder_btn)

        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)

        layout.addLayout(buttons)

    def open_folder(self) -> None:
        folder = os.path.dirname(os.path.abspath(self.image_path))
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
