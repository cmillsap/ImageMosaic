"""
Result Viewer Module

Zoomable views for the main window's canvas: the guide photo with the tile
grid drawn over it, and the finished mosaic.

A mosaic is only interesting at two scales: from across the room, where the
guide image appears, and nose-to-the-glass, where the individual photos do.
Both views open fitted to the space and zoom with the mouse wheel down to
actual pixels, panning by drag.
"""

import math
import os

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import (QColor, QDesktopServices, QImageReader, QPainter,
                         QPen, QPixmap, QTransform)
from PyQt6.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel, QPushButton,
                             QStackedWidget, QVBoxLayout, QWidget)

# Wheel step and the zoom range it is clamped to, as a scale factor where
# 1.0 is one image pixel per screen pixel.
ZOOM_STEP = 1.25
MAX_ZOOM = 8.0

# Grid lines closer together than this on screen blur into a flat tint, so
# below it only the outline is drawn.
MIN_GRID_SPACING_PX = 4


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

    def __init__(self, pixmap: QPixmap = None, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        # Stays in "fit" mode, refitting on resize, until the user zooms.
        self._fitted = True
        if pixmap is not None:
            self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap, size=None) -> None:
        """Show pixmap, stretched to size (width, height) in scene units.

        Without a size the pixmap is shown at its own dimensions.
        """
        self.item.setPixmap(pixmap)
        transform = QTransform()
        if size is not None and not pixmap.isNull():
            transform = QTransform.fromScale(size[0] / pixmap.width(),
                                             size[1] / pixmap.height())
        self.item.setTransform(transform)
        # The scene only ever grows on its own; pin it to the new image.
        self.scene().setSceneRect(self.item.sceneBoundingRect())
        self.fit()

    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def min_zoom(self) -> float:
        """The zoom at which the whole image fits; never zoom out past it."""
        rect = self.item.sceneBoundingRect()
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


class GridPreview(ZoomableImageView):
    """The guide photo as the renderer sees it, with the tile grid on top.

    Scene units are output pixels. The photo is stretched over the whole
    grid, exactly as GuideImage stretches it before cutting cells, so what
    lands in each drawn cell here is what that cell is matched against.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.cols = 0
        self.rows = 0
        self.tile_width = 0
        self.tile_height = 0
        self.show_grid = True
        self._photo = QPixmap()

    def set_photo(self, pixmap: QPixmap) -> None:
        self._photo = pixmap
        self._refresh()

    def set_grid(self, cols: int, rows: int,
                 tile_width: int, tile_height: int) -> None:
        self.cols, self.rows = cols, rows
        self.tile_width, self.tile_height = tile_width, tile_height
        self._refresh()

    def set_show_grid(self, show: bool) -> None:
        self.show_grid = show
        self.viewport().update()

    def _refresh(self) -> None:
        if self._photo.isNull() or not (self.cols and self.rows):
            self.set_pixmap(self._photo)
        else:
            self.set_pixmap(self._photo, (self.cols * self.tile_width,
                                          self.rows * self.tile_height))
        self.viewport().update()

    def drawForeground(self, painter, rect):
        if (not self.show_grid or self._photo.isNull()
                or not (self.cols and self.rows)):
            return
        width = self.cols * self.tile_width
        height = self.rows * self.tile_height

        pen = QPen(QColor(255, 255, 255, 110))
        pen.setCosmetic(True)  # one screen pixel wide at any zoom
        painter.setPen(pen)

        spacing = min(self.tile_width, self.tile_height) * self.zoom
        if spacing >= MIN_GRID_SPACING_PX:
            # Only the lines inside the exposed area; a fine grid on a large
            # print can run to thousands.
            first_col = max(1, math.floor(rect.left() / self.tile_width))
            last_col = min(self.cols - 1, math.ceil(rect.right() / self.tile_width))
            for c in range(first_col, last_col + 1):
                x = c * self.tile_width
                painter.drawLine(x, 0, x, height)
            first_row = max(1, math.floor(rect.top() / self.tile_height))
            last_row = min(self.rows - 1, math.ceil(rect.bottom() / self.tile_height))
            for r in range(first_row, last_row + 1):
                y = r * self.tile_height
                painter.drawLine(0, y, width, y)

        painter.drawRect(0, 0, width, height)


class ResultPanel(QWidget):
    """Shows the finished mosaic with its summary and file actions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        # The view while an image is showing, else None.
        self.view = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view = ZoomableImageView(parent=self)
        self._message = QLabel("Generate a mosaic to see it here.")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._message)
        self._stack.addWidget(self._view)
        layout.addWidget(self._stack, 1)

        self.info_label = QLabel()
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.info_label.setWordWrap(True)
        self.info_label.setContentsMargins(8, 4, 8, 0)
        self.info_label.hide()
        layout.addWidget(self.info_label)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(8, 4, 8, 4)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self._view.fit)
        buttons.addWidget(self.fit_btn)

        self.actual_btn = QPushButton("100%")
        self.actual_btn.setToolTip("One mosaic pixel per screen pixel")
        self.actual_btn.clicked.connect(self._view.actual_size)
        buttons.addWidget(self.actual_btn)

        hint = QLabel("Scroll to zoom, drag to pan")
        hint.setEnabled(False)  # palette-aware secondary text
        buttons.addWidget(hint)
        buttons.addStretch()

        self.folder_btn = QPushButton("Open Folder")
        self.folder_btn.clicked.connect(self.open_folder)
        buttons.addWidget(self.folder_btn)
        layout.addLayout(buttons)

        self._set_controls_enabled(False)

    def show_result(self, image_path: str, summary: str = "") -> None:
        self.image_path = image_path
        pixmap = load_pixmap(image_path)
        if pixmap.isNull():
            self.view = None
            self._message.setText(f"The mosaic was saved, but could not be "
                                  f"displayed:\n{image_path}")
            self._stack.setCurrentWidget(self._message)
        else:
            self.view = self._view
            self._view.set_pixmap(pixmap)
            self._stack.setCurrentWidget(self._view)

        self.info_label.setText(f"Saved to: {image_path}"
                                + (f"\n{summary}" if summary else ""))
        self.info_label.show()
        self._set_controls_enabled(self.view is not None)
        self.folder_btn.setEnabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for btn in (self.fit_btn, self.actual_btn, self.folder_btn):
            btn.setEnabled(enabled)

    def open_folder(self) -> None:
        if not self.image_path:
            return
        folder = os.path.dirname(os.path.abspath(self.image_path))
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
