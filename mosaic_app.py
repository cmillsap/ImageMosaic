import multiprocessing
import os
import sys
import tempfile

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QLineEdit,
                              QFileDialog, QSpinBox, QCheckBox, QProgressBar,
                              QMessageBox, QComboBox, QSplitter, QScrollArea,
                              QTabWidget, QStackedWidget, QSlider, QToolButton,
                              QButtonGroup, QFrame)
from PyQt6.QtCore import Qt, QThread, QStandardPaths, QSettings
from PyQt6.QtGui import QImage, QPixmap

from image_io import open_image
from mosaic_worker import MosaicJob, MosaicWorker
from result_viewer import GridPreview, ResultPanel
from tile_database import find_tile_files
from tile_preprocessor import PreprocessorConfig

# (label, extension) for each output format offered in the UI.
OUTPUT_FORMATS = [("PNG", ".png"), ("JPEG", ".jpg"), ("TIFF", ".tif")]

# Common print sizes in inches, short side first. Orientation is chosen
# separately, so each size is listed once.
PRINT_SIZES = [(8, 10), (11, 14), (16, 20), (18, 24), (20, 30), (24, 36)]

# Tile shapes offered as buttons, as (label, width part, height part).
TILE_SHAPES = [("1:1", 1, 1), ("4:3", 4, 3), ("3:2", 3, 2)]

# Look presets: named starting points for the four matching controls.
# Any edit to one of those controls drops back to "custom".
LOOK_PRESETS = {
    "Accurate": dict(variety=0, tint=0, min_gap=0, max_uses=0),
    "Balanced": dict(variety=30, tint=15, min_gap=2, max_uses=0),
    "Varied": dict(variety=70, tint=30, min_gap=4, max_uses=0),
}

IMAGE_FILTER = ("Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff "
                "*.heic *.heif *.dng);;All Files (*)")

# Stretching the photo by less than this goes unmentioned.
STRETCH_TOLERANCE = 0.03

# Longest side of the photo kept for the on-screen preview.
PREVIEW_MAX_DIM = 2048

DONE_STYLE = "QLabel { color: #2e8b57; }"
PROBLEM_STYLE = "QLabel { color: #e74c3c; }"


def default_output_folder() -> str:
    """The user's Documents folder, or their home folder if there is none."""
    documents = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.DocumentsLocation)
    return documents or os.path.expanduser("~")


def unique_path(folder: str, stem: str, ext: str) -> str:
    """folder/stem+ext, numbered if needed so an earlier mosaic is kept."""
    candidate = os.path.join(folder, stem + ext)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem} ({n}){ext}")
        n += 1
    return candidate


def load_preview(path: str):
    """(pixmap, (width, height)) for a photo in any format open_image reads.

    Goes through PIL rather than QPixmap so HEIC and DNG preview too. The
    size returned is the original's; the pixmap is capped at
    PREVIEW_MAX_DIM, which is plenty for a screen.

    Raises whatever the decoder raises for an unreadable file.
    """
    with open_image(path) as img:
        size = img.size
        img = img.convert("RGB")
        img.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM))
        data = img.tobytes()
        qimage = QImage(data, img.width, img.height, 3 * img.width,
                        QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage), size


def _section(title: str):
    """A titled sidebar section: (widget, content layout, status label).

    The status label sits right of the title and is used for the green
    tick on the two required inputs.
    """
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(6)

    header = QHBoxLayout()
    label = QLabel(title.upper())
    font = label.font()
    font.setBold(True)
    font.setPointSizeF(font.pointSizeF() * 0.85)
    label.setFont(font)
    header.addWidget(label)
    header.addStretch()
    status = QLabel()
    header.addWidget(status)
    layout.addLayout(header)
    return box, layout, status


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _secondary(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setEnabled(False)  # palette-aware secondary text
    return label


class MosaicApp(QMainWindow):
    def __init__(self, settings: QSettings = None):
        super().__init__()
        # Preferences that outlive the session. Injectable so tests never
        # touch the real user's settings.
        self.settings = settings or QSettings("ImageMosaic", "ImageMosaic")
        self.guide_image_path = None
        # (width, height) of the guide photo in pixels, once one is chosen.
        self.guide_size = None
        self.tile_folder_path = None
        # Images found in the tile folder; None until a folder is scanned.
        self.tile_count = None
        self.scan_subdirectories = False
        self.output_dpi = 300
        self.output_width_inches = 20
        self.output_height_inches = 30
        self.tile_width = 100
        self.tile_height = 100
        self.output_folder = self.saved_output_folder()
        # Set while a render is in flight; both are None when idle.
        self.worker = None
        self.worker_thread = None
        # Set when the user picks "Custom" so typing a size that happens to
        # match a preset doesn't snap the fields shut mid-edit.
        self._custom_print_size = False
        self._custom_tile_shape = False
        self.init_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def init_ui(self):
        self.setWindowTitle("Image Mosaic Generator")
        self.setGeometry(100, 100, 1200, 800)
        self.setAcceptDrops(True)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_canvas())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setSizes([360, 840])
        self.setCentralWidget(splitter)

        self._build_status_bar()

        # Every widget exists now, so it is safe to populate the readouts.
        self.update_derived_dimensions()
        self.sync_look_preset()
        self.update_output_name()

    def _build_sidebar(self) -> QWidget:
        sections = QWidget()
        column = QVBoxLayout(sections)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        for build in (self._build_photo_section, self._build_tiles_section,
                      self._build_print_section, self._build_tile_size_section,
                      self._build_look_section):
            column.addWidget(build())
            column.addWidget(_divider())
        column.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(sections)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar = QWidget()
        sidebar.setMinimumWidth(320)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(scroll, 1)
        layout.addWidget(_divider())
        # Save and Generate stay pinned below the scrolling settings.
        layout.addWidget(self._build_footer())
        return sidebar

    def _build_photo_section(self) -> QWidget:
        box, layout, self.photo_status_label = _section("Photo")

        row = QHBoxLayout()
        self.guide_thumbnail = QLabel()
        self.guide_thumbnail.setFixedSize(72, 54)
        self.guide_thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.guide_thumbnail.setFrameShape(QFrame.Shape.StyledPanel)
        row.addWidget(self.guide_thumbnail)

        details = QVBoxLayout()
        details.setSpacing(2)
        self.guide_name_label = QLabel("No photo chosen")
        self.guide_name_label.setWordWrap(True)
        details.addWidget(self.guide_name_label)
        self.guide_size_label = _secondary("Or drop one on the preview")
        details.addWidget(self.guide_size_label)
        row.addLayout(details, 1)
        layout.addLayout(row)

        self.select_image_btn = QPushButton("Choose Photo…")
        self.select_image_btn.clicked.connect(self.select_guide_image)
        layout.addWidget(self.select_image_btn)
        return box

    def _build_tiles_section(self) -> QWidget:
        box, layout, self.tiles_status_label = _section("Tile photos")

        row = QHBoxLayout()
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("No folder chosen")
        self.folder_path_edit.setReadOnly(True)
        row.addWidget(self.folder_path_edit)
        browse_btn = QPushButton("Choose…")
        browse_btn.clicked.connect(self.select_tile_folder)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        self.subdirs_checkbox = QCheckBox("Include subfolders")
        self.subdirs_checkbox.setChecked(False)
        self.subdirs_checkbox.stateChanged.connect(self.update_scan_subdirectories)
        layout.addWidget(self.subdirs_checkbox)
        return box

    def _build_print_section(self) -> QWidget:
        box, layout, _ = _section("Print size")

        row = QHBoxLayout()
        self.print_size_combo = QComboBox()
        for short, long in PRINT_SIZES:
            self.print_size_combo.addItem(f"{short} × {long} in", (short, long))
        self.print_size_combo.addItem("Custom…", None)
        self.print_size_combo.currentIndexChanged.connect(self.on_print_size_chosen)
        row.addWidget(self.print_size_combo, 1)

        self.orientation_btn = QPushButton("⇄")
        self.orientation_btn.setToolTip("Swap portrait and landscape")
        self.orientation_btn.setFixedWidth(36)
        self.orientation_btn.clicked.connect(self.swap_orientation)
        row.addWidget(self.orientation_btn)
        layout.addLayout(row)

        # Width and height, shown only for a custom size.
        self.custom_print_row = QWidget()
        custom = QHBoxLayout(self.custom_print_row)
        custom.setContentsMargins(0, 0, 0, 0)
        custom.addWidget(QLabel("Width"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 100)
        self.width_spinbox.setSuffix(" in")
        self.width_spinbox.setValue(self.output_width_inches)
        self.width_spinbox.valueChanged.connect(self.on_print_dimension_edited)
        custom.addWidget(self.width_spinbox)
        custom.addWidget(QLabel("Height"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 100)
        self.height_spinbox.setSuffix(" in")
        self.height_spinbox.setValue(self.output_height_inches)
        self.height_spinbox.valueChanged.connect(self.on_print_dimension_edited)
        custom.addWidget(self.height_spinbox)
        custom.addStretch()
        layout.addWidget(self.custom_print_row)

        self.output_dimensions_label = _secondary()
        layout.addWidget(self.output_dimensions_label)

        # Warns when the photo's shape differs from the print's, since
        # GuideImage stretches it to fit rather than cropping.
        self.stretch_label = QLabel()
        self.stretch_label.setWordWrap(True)
        self.stretch_label.setStyleSheet(PROBLEM_STYLE)
        self.stretch_label.setVisible(False)
        layout.addWidget(self.stretch_label)

        self.sync_print_size()
        return box

    def _build_tile_size_section(self) -> QWidget:
        box, layout, _ = _section("Tiles")

        row = QHBoxLayout()
        self.tile_size_label = QLabel("Size")
        row.addWidget(self.tile_size_label)
        self.tile_width_spinbox = QSpinBox()
        self.tile_width_spinbox.setRange(8, 2000)
        self.tile_width_spinbox.setSuffix(" px")
        self.tile_width_spinbox.setValue(self.tile_width)
        self.tile_width_spinbox.valueChanged.connect(self.on_tile_width_edited)
        row.addWidget(self.tile_width_spinbox)
        row.addStretch()

        # Exclusive, checkable buttons: one per shape plus Custom.
        self.tile_shape_group = QButtonGroup(self)
        self.tile_shape_buttons = []
        for i, (label, _, _) in enumerate(TILE_SHAPES + [("Custom", 0, 0)]):
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            self.tile_shape_group.addButton(btn, i)
            self.tile_shape_buttons.append(btn)
            row.addWidget(btn)
        self.tile_shape_buttons[0].setChecked(True)
        self.tile_shape_group.idClicked.connect(self.on_tile_shape_chosen)
        layout.addLayout(row)

        # Height, shown only for a custom shape.
        self.custom_tile_row = QWidget()
        custom = QHBoxLayout(self.custom_tile_row)
        custom.setContentsMargins(0, 0, 0, 0)
        custom.addWidget(QLabel("Height"))
        self.tile_height_spinbox = QSpinBox()
        self.tile_height_spinbox.setRange(8, 2000)
        self.tile_height_spinbox.setSuffix(" px")
        self.tile_height_spinbox.setValue(self.tile_height)
        self.tile_height_spinbox.valueChanged.connect(self.on_tile_height_edited)
        custom.addWidget(self.tile_height_spinbox)
        custom.addStretch()
        self.custom_tile_row.setVisible(False)
        layout.addWidget(self.custom_tile_row)

        # Aspect ratio readout - tiles are cropped to this ratio by the
        # preprocessor, so it is also the shape of every guide grid cell.
        self.tile_aspect_label = _secondary()
        layout.addWidget(self.tile_aspect_label)

        self.grid_info_label = QLabel()
        self.grid_info_label.setWordWrap(True)
        layout.addWidget(self.grid_info_label)
        return box

    def _build_look_section(self) -> QWidget:
        box, layout, self.look_custom_label = _section("Look")
        self.look_custom_label.setEnabled(False)

        presets = QHBoxLayout()
        presets.setSpacing(0)
        self.look_group = QButtonGroup(self)
        self.look_buttons = {}
        tips = {
            "Accurate": "Closest colour match. Expect repeats in flat areas.",
            "Balanced": "Spreads repeats out, with a light tint to keep colours true.",
            "Varied": "Shows as much of your library as possible.",
        }
        for name in LOOK_PRESETS:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setToolTip(tips[name])
            btn.clicked.connect(lambda _, n=name: self.apply_look_preset(n))
            self.look_group.addButton(btn)
            self.look_buttons[name] = btn
            presets.addWidget(btn)
        layout.addLayout(presets)

        # Soft variety and colour tint. Unlike the hard limits under
        # Advanced, these trade colour accuracy for variety gradually.
        self.variety_slider, self.variety_value_label = self._add_slider(
            layout, "Variety",
            "Favour images that have been used less.\n"
            "Each use makes an image slightly less likely to be picked again,\n"
            "so more of your library appears at some cost to colour accuracy.",
            suffix="")
        self.tint_slider, self.tint_value_label = self._add_slider(
            layout, "Tint",
            "Shift each image's colours toward the part of the guide it covers.\n"
            "Makes loosely matched images read correctly, so Variety can be\n"
            "raised without the mosaic going muddy. 15-30% is usually subtle.",
            suffix="%")

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        self.advanced_toggle.toggled.connect(self.set_advanced_visible)
        layout.addWidget(self.advanced_toggle)

        self.advanced_panel = QWidget()
        advanced = QVBoxLayout(self.advanced_panel)
        advanced.setContentsMargins(16, 0, 0, 0)

        # Hard variety limits. Without these a plain nearest-neighbour match
        # reuses one photo across every flat area of the guide.
        row = QHBoxLayout()
        row.addWidget(QLabel("Max uses per image"))
        row.addStretch()
        self.max_reuse_spinbox = QSpinBox()
        self.max_reuse_spinbox.setRange(0, 9999)
        self.max_reuse_spinbox.setValue(0)
        self.max_reuse_spinbox.setSpecialValueText("unlimited")
        self.max_reuse_spinbox.setToolTip(
            "Cap how many times one image may appear. 0 means no limit.\n"
            "Best effort: if no alternative fits, the least-used image wins."
        )
        self.max_reuse_spinbox.valueChanged.connect(self.sync_look_preset)
        row.addWidget(self.max_reuse_spinbox)
        advanced.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min gap between repeats"))
        row.addStretch()
        self.min_distance_spinbox = QSpinBox()
        self.min_distance_spinbox.setRange(0, 50)
        self.min_distance_spinbox.setValue(0)
        self.min_distance_spinbox.setSpecialValueText("none")
        self.min_distance_spinbox.setSuffix(" cells")
        self.min_distance_spinbox.setToolTip(
            "Keep repeats of the same image at least this many cells apart."
        )
        self.min_distance_spinbox.valueChanged.connect(self.sync_look_preset)
        row.addWidget(self.min_distance_spinbox)
        advanced.addLayout(row)

        self.randomize_order_checkbox = QCheckBox("Randomize placement order")
        self.randomize_order_checkbox.setChecked(True)
        self.randomize_order_checkbox.setToolTip(
            "Fill cells in a shuffled order rather than row by row. When\n"
            "Variety or a limit is on, this spreads the compromise evenly\n"
            "instead of leaving the bottom rows with the leftover images."
        )
        advanced.addWidget(self.randomize_order_checkbox)

        self.advanced_panel.setVisible(False)
        layout.addWidget(self.advanced_panel)
        return box

    def _add_slider(self, layout, label, tooltip, suffix):
        """A labelled 0-100 slider with its value shown alongside."""
        row = QHBoxLayout()
        name = QLabel(label)
        name.setFixedWidth(52)
        name.setToolTip(tooltip)
        row.addWidget(name)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(0)
        slider.setToolTip(tooltip)
        row.addWidget(slider, 1)
        value = QLabel()
        value.setFixedWidth(36)
        value.setAlignment(Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value)
        layout.addLayout(row)

        def show_value(v):
            value.setText(f"{v}{suffix}" if v else "off")
        slider.valueChanged.connect(show_value)
        slider.valueChanged.connect(self.sync_look_preset)
        show_value(0)
        return slider, value

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("Save to"))
        self.output_folder_edit = QLineEdit(self.output_folder)
        self.output_folder_edit.setReadOnly(True)
        row.addWidget(self.output_folder_edit, 1)
        output_browse_btn = QPushButton("Change…")
        output_browse_btn.clicked.connect(self.select_output_folder)
        row.addWidget(output_browse_btn)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("As"))
        self.output_name_label = _secondary()
        row.addWidget(self.output_name_label, 1)
        self.output_format_combo = QComboBox()
        for label, ext in OUTPUT_FORMATS:
            self.output_format_combo.addItem(label, ext)
        self.output_format_combo.currentIndexChanged.connect(self.update_output_name)
        row.addWidget(self.output_format_combo)
        layout.addLayout(row)

        self.generate_btn = QPushButton("Generate Mosaic")
        self.generate_btn.setEnabled(False)
        self.generate_btn.setDefault(True)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_mosaic)
        layout.addWidget(self.generate_btn)

        # Says why Generate is disabled; empty when it isn't.
        self.generate_hint_label = _secondary()
        self.generate_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.generate_hint_label.setWordWrap(True)
        layout.addWidget(self.generate_hint_label)
        return footer

    def _build_canvas(self) -> QWidget:
        self.canvas_tabs = QTabWidget()
        self.canvas_tabs.setDocumentMode(True)

        photo_tab = QWidget()
        layout = QVBoxLayout(photo_tab)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_stack = QStackedWidget()
        self.preview_placeholder = QLabel(
            "Drop a photo here, or choose one on the left.\n\n"
            "Drop a folder to use it for the tile photos.")
        self.preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_placeholder.setEnabled(False)
        self.preview_stack.addWidget(self.preview_placeholder)
        self.grid_preview = GridPreview()
        # Let drops fall through to the window, which knows what to do.
        self.grid_preview.setAcceptDrops(False)
        self.grid_preview.viewport().setAcceptDrops(False)
        self.preview_stack.addWidget(self.grid_preview)
        layout.addWidget(self.preview_stack, 1)

        self.preview_controls = QWidget()
        controls = QHBoxLayout(self.preview_controls)
        controls.setContentsMargins(8, 4, 8, 4)
        self.show_grid_checkbox = QCheckBox("Show tile grid")
        self.show_grid_checkbox.setChecked(True)
        self.show_grid_checkbox.toggled.connect(self.grid_preview.set_show_grid)
        controls.addWidget(self.show_grid_checkbox)
        controls.addStretch()
        controls.addWidget(_secondary("Scroll to zoom, drag to pan"))
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.grid_preview.fit)
        controls.addWidget(fit_btn)
        actual_btn = QPushButton("100%")
        actual_btn.setToolTip("One print pixel per screen pixel")
        actual_btn.clicked.connect(self.grid_preview.actual_size)
        controls.addWidget(actual_btn)
        self.preview_controls.setEnabled(False)  # nothing to show yet
        layout.addWidget(self.preview_controls)

        self.result_panel = ResultPanel()

        self.canvas_tabs.addTab(photo_tab, "Photo + grid")
        self.canvas_tabs.addTab(self.result_panel, "Mosaic")
        self.canvas_tabs.setTabEnabled(1, False)
        return self.canvas_tabs

    def _build_status_bar(self):
        bar = self.statusBar()
        self.progress_status_label = QLabel("Ready")
        bar.addWidget(self.progress_status_label, 1)

        # Progress bar and Cancel, shown only while a render runs.
        self.progress_widget = QWidget()
        layout = QHBoxLayout(self.progress_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(260)
        layout.addWidget(self.progress_bar)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_mosaic)
        self.cancel_btn.setVisible(False)
        layout.addWidget(self.cancel_btn)
        self.progress_widget.setVisible(False)
        bar.addPermanentWidget(self.progress_widget)

    # ------------------------------------------------------------------
    # Photo and tile folder
    # ------------------------------------------------------------------

    def select_guide_image(self):
        """Open file dialog to select guide image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose Photo", "", IMAGE_FILTER)
        if file_path:
            self.set_guide_image(file_path)

    def set_guide_image(self, file_path: str) -> bool:
        """Use file_path as the guide photo. False if it can't be read."""
        try:
            pixmap, (width, height) = load_preview(file_path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Couldn't open photo",
                f"{os.path.basename(file_path)} couldn't be read as an "
                f"image.\n\n{exc}")
            return False

        self.guide_image_path = file_path
        self.guide_size = (width, height)
        self.guide_thumbnail.setPixmap(pixmap.scaled(
            self.guide_thumbnail.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.guide_name_label.setText(os.path.basename(file_path))
        self.guide_size_label.setText(f"{width} × {height} px")
        self.select_image_btn.setText("Change Photo…")
        self.photo_status_label.setText("✓")
        self.photo_status_label.setStyleSheet(DONE_STYLE)

        self.grid_preview.set_photo(pixmap)
        self.preview_stack.setCurrentWidget(self.grid_preview)
        self.preview_controls.setEnabled(True)
        self.canvas_tabs.setCurrentIndex(0)
        self.update_output_name()
        self.update_derived_dimensions()
        return True

    def select_tile_folder(self):
        """Open folder dialog to select tile images folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Choose Tile Photos Folder", self.tile_folder_path or "")
        if folder_path:
            self.set_tile_folder(folder_path)

    def set_tile_folder(self, folder_path: str):
        self.tile_folder_path = folder_path
        self.folder_path_edit.setText(folder_path)
        self.refresh_tile_count()

    def update_scan_subdirectories(self, state):
        """Update the scan subdirectories setting based on checkbox state"""
        self.scan_subdirectories = state == Qt.CheckState.Checked.value
        self.refresh_tile_count()

    def refresh_tile_count(self):
        """Count the images in the tile folder and show it in the header.

        Only lists file names, so it stays quick even for a large library;
        nothing is decoded until Generate.
        """
        if not self.tile_folder_path:
            self.tile_count = None
            self.tiles_status_label.setText("")
        else:
            try:
                self.tile_count = len(find_tile_files(
                    self.tile_folder_path, recursive=self.scan_subdirectories))
            except (OSError, ValueError):
                self.tile_count = None
                self.tiles_status_label.setText("Folder not found")
                self.tiles_status_label.setStyleSheet(PROBLEM_STYLE)
            else:
                if self.tile_count:
                    self.tiles_status_label.setText(
                        f"✓ {self.tile_count:,} found")
                    self.tiles_status_label.setStyleSheet(DONE_STYLE)
                else:
                    self.tiles_status_label.setText("No images found")
                    self.tiles_status_label.setStyleSheet(PROBLEM_STYLE)
        self.check_ready_to_generate()

    def dragEnterEvent(self, event):
        if self.worker_thread is None and self._dropped_path(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._dropped_path(event)
        if not path:
            return
        event.acceptProposedAction()
        if os.path.isdir(path):
            self.set_tile_folder(path)
        else:
            self.set_guide_image(path)

    @staticmethod
    def _dropped_path(event):
        """The first local file or folder being dragged, else None."""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            return urls[0].toLocalFile()
        return None

    # ------------------------------------------------------------------
    # Print size
    # ------------------------------------------------------------------

    def on_print_size_chosen(self, index: int):
        size = self.print_size_combo.itemData(index)
        if size is None:
            self._custom_print_size = True
            self.custom_print_row.setVisible(True)
            return
        self._custom_print_size = False
        short, long = size
        # Keep whichever orientation is current.
        if self.width_spinbox.value() > self.height_spinbox.value():
            short, long = long, short
        self._set_print_inches(short, long)

    def on_print_dimension_edited(self):
        self.sync_print_size()
        self.update_derived_dimensions()

    def swap_orientation(self):
        self._set_print_inches(self.height_spinbox.value(),
                               self.width_spinbox.value())

    def _set_print_inches(self, width: int, height: int):
        for box, value in ((self.width_spinbox, width),
                           (self.height_spinbox, height)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self.sync_print_size()
        self.update_derived_dimensions()

    def sync_print_size(self):
        """Point the dropdown at the preset matching the inch fields.

        Falls back to Custom, which reveals the fields, when none matches.
        """
        dims = tuple(sorted((self.width_spinbox.value(),
                             self.height_spinbox.value())))
        # Items are in PRINT_SIZES order. findData can't match a tuple.
        index = PRINT_SIZES.index(dims) if dims in PRINT_SIZES else -1
        if index < 0 or self._custom_print_size:
            self._custom_print_size = True
            index = self.print_size_combo.count() - 1
        self.print_size_combo.blockSignals(True)
        self.print_size_combo.setCurrentIndex(index)
        self.print_size_combo.blockSignals(False)
        self.custom_print_row.setVisible(self._custom_print_size)

    # ------------------------------------------------------------------
    # Tile shape
    # ------------------------------------------------------------------

    def on_tile_shape_chosen(self, shape_id: int):
        if shape_id >= len(TILE_SHAPES):
            self._set_tile_custom(True)
            return
        self._set_tile_custom(False)
        self._derive_tile_height()
        self.update_derived_dimensions()

    def on_tile_width_edited(self):
        if not self._custom_tile_shape:
            self._derive_tile_height()
        self.update_derived_dimensions()

    def on_tile_height_edited(self):
        if not self._custom_tile_shape:
            # Height was set directly; follow it to whichever shape fits.
            shape = self._matching_tile_shape()
            if shape is None:
                self._set_tile_custom(True)
            else:
                self.tile_shape_buttons[shape].setChecked(True)
        self.update_derived_dimensions()

    def _derive_tile_height(self):
        """Set the height from the width and the checked shape."""
        shape = self.tile_shape_group.checkedId()
        if not 0 <= shape < len(TILE_SHAPES):
            return
        _, rw, rh = TILE_SHAPES[shape]
        height = max(8, round(self.tile_width_spinbox.value() * rh / rw))
        self.tile_height_spinbox.blockSignals(True)
        self.tile_height_spinbox.setValue(height)
        self.tile_height_spinbox.blockSignals(False)

    def _matching_tile_shape(self):
        width = self.tile_width_spinbox.value()
        height = self.tile_height_spinbox.value()
        for i, (_, rw, rh) in enumerate(TILE_SHAPES):
            if round(width * rh / rw) == height:
                return i
        return None

    def _set_tile_custom(self, custom: bool):
        self._custom_tile_shape = custom
        if custom:
            self.tile_shape_buttons[-1].setChecked(True)
        self.custom_tile_row.setVisible(custom)
        self.tile_size_label.setText("Width" if custom else "Size")

    # ------------------------------------------------------------------
    # Look
    # ------------------------------------------------------------------

    def apply_look_preset(self, name: str):
        values = LOOK_PRESETS[name]
        controls = ((self.variety_slider, values["variety"]),
                    (self.tint_slider, values["tint"]),
                    (self.min_distance_spinbox, values["min_gap"]),
                    (self.max_reuse_spinbox, values["max_uses"]))
        for control, value in controls:
            control.setValue(value)
        self.randomize_order_checkbox.setChecked(True)
        self.sync_look_preset()

    def current_look_preset(self):
        """The preset the matching controls currently equal, else None."""
        current = dict(variety=self.variety_slider.value(),
                       tint=self.tint_slider.value(),
                       min_gap=self.min_distance_spinbox.value(),
                       max_uses=self.max_reuse_spinbox.value())
        for name, values in LOOK_PRESETS.items():
            if values == current:
                return name
        return None

    def sync_look_preset(self):
        """Check the preset button the controls match, or show Custom."""
        if not hasattr(self, "look_buttons"):
            return  # Still building the section.
        name = self.current_look_preset()
        # An exclusive group can't have nothing checked, so lift the rule
        # while clearing it.
        self.look_group.setExclusive(False)
        for preset, btn in self.look_buttons.items():
            btn.setChecked(preset == name)
        self.look_group.setExclusive(True)
        self.look_custom_label.setText("" if name else "Custom")

    def set_advanced_visible(self, visible: bool):
        self.advanced_panel.setVisible(visible)
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    # ------------------------------------------------------------------
    # Derived sizes
    # ------------------------------------------------------------------

    def update_derived_dimensions(self):
        """Recompute output pixel size and the tile grid from the spin boxes.

        Single entry point for every size control, so the pixel readout, the
        tile aspect ratio, the grid summary and the preview overlay can never
        drift apart.
        """
        self.output_width_inches = self.width_spinbox.value()
        self.output_height_inches = self.height_spinbox.value()
        self.tile_width = self.tile_width_spinbox.value()
        self.tile_height = self.tile_height_spinbox.value()

        w = self.output_width_px
        h = self.output_height_px
        orientation = ("square" if w == h
                       else "landscape" if w > h else "portrait")
        self.output_dimensions_label.setText(
            f"{w} × {h} px at {self.output_dpi} DPI · {orientation}"
        )

        self.tile_aspect_label.setText(
            f"{self.tile_width} × {self.tile_height} px · "
            f"aspect ratio {self.tile_aspect_ratio:.3f}"
        )

        self.update_stretch_warning()

        cols, rows = self.grid_cols, self.grid_rows
        self.grid_preview.set_grid(cols, rows, self.tile_width, self.tile_height)
        if cols == 0 or rows == 0:
            self.grid_info_label.setText(
                "⚠ Tile is larger than the print - no tiles fit."
            )
            self.grid_info_label.setStyleSheet(PROBLEM_STYLE)
        else:
            rem_x, rem_y = self.remainder_px
            text = f"{cols} × {rows} grid = {self.total_tiles:,} tiles"
            if rem_x or rem_y:
                text += f"\n{rem_x} × {rem_y} px unused at edges"
            self.grid_info_label.setText(text)
            self.grid_info_label.setStyleSheet("")
        self.check_ready_to_generate()

    def update_stretch_warning(self):
        """Say how far the photo will be stretched to fill the grid."""
        stretch = self.photo_stretch
        if stretch is None or abs(stretch - 1) < STRETCH_TOLERANCE:
            self.stretch_label.setVisible(False)
            return
        if stretch > 1:
            text = f"wider by {stretch - 1:.0%}"
        else:
            text = f"taller by {1 / stretch - 1:.0%}"
        self.stretch_label.setText(
            f"⚠ The photo will be stretched {text} to fill this print. "
            f"Try ⇄ or a size closer to its shape.")
        self.stretch_label.setVisible(True)

    @property
    def photo_stretch(self):
        """Horizontal stretch GuideImage applies to the photo, relative to
        vertical: 1.0 means undistorted. None without a photo or grid."""
        if not self.guide_size or not self.total_tiles:
            return None
        used_w = self.grid_cols * self.tile_width
        used_h = self.grid_rows * self.tile_height
        photo_w, photo_h = self.guide_size
        return (used_w / used_h) / (photo_w / photo_h)

    @property
    def output_width_px(self):
        return self.output_width_inches * self.output_dpi

    @property
    def output_height_px(self):
        return self.output_height_inches * self.output_dpi

    @property
    def tile_aspect_ratio(self):
        """Width / height of a single tile."""
        return self.tile_width / self.tile_height

    @property
    def grid_cols(self):
        """Number of whole tiles that fit across the output."""
        return self.output_width_px // self.tile_width

    @property
    def grid_rows(self):
        """Number of whole tiles that fit down the output."""
        return self.output_height_px // self.tile_height

    @property
    def total_tiles(self):
        """Total tile placements in the mosaic."""
        return self.grid_cols * self.grid_rows

    @property
    def remainder_px(self):
        """(x, y) pixels left over that no whole tile covers."""
        return (self.output_width_px - self.grid_cols * self.tile_width,
                self.output_height_px - self.grid_rows * self.tile_height)

    def build_preprocessor_config(self, **overrides) -> PreprocessorConfig:
        """Build a PreprocessorConfig from the UI's tile dimensions.

        The preprocessor crops tiles to this aspect ratio and GuideImage cuts
        grid cells at these same dimensions. Deriving both from one place is
        what keeps them in agreement - a mismatch would distort every tile.
        """
        params = dict(target_width=self.tile_width,
                      target_height=self.tile_height)
        params.update(overrides)
        return PreprocessorConfig(**params)

    # ------------------------------------------------------------------
    # Output location
    # ------------------------------------------------------------------

    def select_output_folder(self):
        """Open folder dialog to choose where mosaics are saved"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Choose Output Folder",
            self.output_folder
        )

        if folder_path:
            self.output_folder = folder_path
            self.output_folder_edit.setText(folder_path)
            self.settings.setValue("output_folder", folder_path)

    def saved_output_folder(self) -> str:
        """The last folder chosen with Browse, else Documents.

        Falls back if the saved folder has since been deleted or was on a
        drive that is no longer attached.
        """
        folder = self.settings.value("output_folder", "", type=str)
        if folder and os.path.isdir(folder):
            return folder
        return default_output_folder()

    def output_stem(self) -> str:
        """<guide name>_mosaic, or plain "mosaic" before a photo is chosen."""
        if not self.guide_image_path:
            return "mosaic"
        stem = os.path.splitext(os.path.basename(self.guide_image_path))[0]
        return f"{stem}_mosaic"

    def update_output_name(self):
        self.output_name_label.setText(
            self.output_stem() + self.output_format_combo.currentData())

    def check_ready_to_generate(self):
        """Enable Generate only when a run could start, else say why not."""
        if not hasattr(self, "generate_btn"):
            return  # Still building the window.
        if not self.guide_image_path and not self.tile_folder_path:
            reason = "Choose a photo and a folder of tile photos to start."
        elif not self.guide_image_path:
            reason = "Choose a photo to turn into a mosaic."
        elif not self.tile_folder_path:
            reason = "Choose a folder of tile photos."
        elif self.tile_count == 0:
            reason = ("No images in that folder. Try including subfolders "
                      "or choose another.")
        elif self.total_tiles == 0:
            reason = "Make the tiles smaller or the print larger."
        else:
            reason = ""
        running = self.worker_thread is not None
        self.generate_btn.setEnabled(not reason and not running)
        self.generate_hint_label.setText(reason)
        self.generate_hint_label.setVisible(bool(reason))

    # Progress UI Methods

    def show_progress(self):
        """Show the progress section and prepare for a new task."""
        self.progress_widget.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_status_label.setText("Starting...")
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setVisible(True)

    def hide_progress(self):
        """Hide the progress section and reset to ready state."""
        self.progress_widget.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_status_label.setText("Ready")
        self.cancel_btn.setVisible(False)
        self.check_ready_to_generate()

    def set_progress_status(self, status: str):
        """Set the current task status text."""
        self.progress_status_label.setText(status)

    def set_progress_value(self, value: int):
        """Set the progress bar value (0-100)."""
        self.progress_bar.setValue(max(0, min(100, value)))

    def set_progress(self, status: str, value: int):
        """Set both status text and progress value at once."""
        self.set_progress_status(status)
        self.set_progress_value(value)

    # Mosaic Generation

    def build_job(self, output_path: str) -> MosaicJob:
        """Snapshot the current settings into a job for the worker.

        Read on the GUI thread before the worker starts, so the worker never
        reads a control the user might still be editing.
        """
        return MosaicJob(
            guide_image_path=self.guide_image_path,
            tile_folder_path=self.tile_folder_path,
            output_path=output_path,
            tile_width=self.tile_width,
            tile_height=self.tile_height,
            output_width_px=self.output_width_px,
            output_height_px=self.output_height_px,
            output_dpi=self.output_dpi,
            scan_subdirectories=self.scan_subdirectories,
            max_tile_reuse=self.max_reuse_spinbox.value(),
            min_reuse_distance=self.min_distance_spinbox.value(),
            variety=self.variety_slider.value(),
            randomize_order=self.randomize_order_checkbox.isChecked(),
            tint_strength=self.tint_slider.value(),
            cache_dir=self.tile_cache_dir(),
        )

    def tile_cache_dir(self) -> str:
        """Where preprocessed tiles are cached between runs.

        Tiles are prepared twice - once to index their colours and again to
        paste their pixels - so an on-disk cache roughly halves the work and
        makes a second run over the same library far faster.
        """
        return os.path.join(tempfile.gettempdir(), "image_mosaic_tile_cache")

    def choose_output_path(self):
        """The file to save to: <guide name>_mosaic in the output folder.

        Never overwrites - a repeat run gets a numbered name instead.
        """
        ext = self.output_format_combo.currentData()
        return unique_path(self.output_folder, self.output_stem(), ext)

    def generate_mosaic(self):
        """Validate, pick an output path, and start the worker thread."""
        if self.worker_thread is not None:
            return  # Already running.

        if self.total_tiles == 0:
            QMessageBox.warning(
                self, "Nothing to generate",
                "The tile is larger than the output canvas, so no tiles fit. "
                "Reduce the tile size or increase the output dimensions."
            )
            return

        output_path = self.choose_output_path()
        if not output_path:
            return

        job = self.build_job(output_path)

        self.worker = MosaicWorker(job)
        self.worker_thread = QThread(self)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.stage_progress.connect(self.on_stage_progress)
        self.worker.overall_progress.connect(self.set_progress_value)
        self.worker.finished.connect(self.on_render_finished)
        self.worker.failed.connect(self.on_render_failed)
        self.worker.cancelled.connect(self.on_render_cancelled)

        self.show_progress()
        self.worker_thread.start()

    def cancel_mosaic(self):
        """Ask the running worker to stop at its next checkpoint."""
        if self.worker is not None:
            self.cancel_btn.setEnabled(False)
            self.set_progress_status("Cancelling...")
            self.worker.cancel()

    def on_stage_progress(self, stage: str, completed: int, total: int):
        """Show which stage is running and how far through it is."""
        if total > 1:
            self.set_progress_status(f"{stage}  ({completed:,} of {total:,})")
        else:
            self.set_progress_status(stage)

    def on_render_finished(self, output_path: str, stats):
        self._teardown_worker()
        self.hide_progress()
        self.show_result(
            output_path,
            f"{stats.total_cells:,} tiles placed, "
            f"{stats.distinct_tiles:,} distinct images used. "
            f"Most-used image appears {stats.max_tile_uses:,} times."
        )
        self.progress_status_label.setText(
            f"Saved {os.path.basename(output_path)}")

    def show_result(self, output_path: str, summary: str):
        """Show the finished mosaic in the canvas's Mosaic tab."""
        self.result_panel.show_result(output_path, summary)
        self.canvas_tabs.setTabEnabled(1, True)
        self.canvas_tabs.setCurrentIndex(1)

    def on_render_failed(self, message: str):
        self._teardown_worker()
        self.hide_progress()
        QMessageBox.critical(self, "Mosaic failed", message)

    def on_render_cancelled(self):
        self._teardown_worker()
        self.hide_progress()
        self.set_progress_status("Cancelled")

    def _teardown_worker(self):
        """Stop the thread and drop both objects."""
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread.deleteLater()
            self.worker_thread = None
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def closeEvent(self, event):
        """Never leave a worker thread running after the window closes."""
        if self.worker is not None:
            self.worker.cancel()
        self._teardown_worker()
        super().closeEvent(event)


def main():
    # Frozen builds re-execute this file to start each tile-analysis worker
    # process. Without freeze_support() every worker would spawn its own
    # window instead of doing the work.
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    window = MosaicApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
