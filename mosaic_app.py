import multiprocessing
import os
import sys
import tempfile

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QLineEdit,
                              QFileDialog, QGroupBox, QSizePolicy, QSpinBox,
                              QCheckBox, QProgressBar, QMessageBox,
                              QComboBox)
from PyQt6.QtCore import Qt, QThread, QStandardPaths, QSettings
from PyQt6.QtGui import QPixmap

from mosaic_worker import MosaicJob, MosaicWorker
from result_viewer import ResultViewer
from tile_preprocessor import PreprocessorConfig

# (label, extension) for each output format offered in the UI.
OUTPUT_FORMATS = [("PNG", ".png"), ("JPEG", ".jpg"), ("TIFF", ".tif")]


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


class MosaicApp(QMainWindow):
    def __init__(self, settings: QSettings = None):
        super().__init__()
        # Preferences that outlive the session. Injectable so tests never
        # touch the real user's settings.
        self.settings = settings or QSettings("ImageMosaic", "ImageMosaic")
        self.guide_image_path = None
        self.tile_folder_path = None
        self.scan_subdirectories = False
        self.output_dpi = 300
        self.output_width_inches = 20
        self.output_height_inches = 30
        self.tile_width = 100
        self.tile_height = 100
        self.output_folder = self.saved_output_folder()
        self.result_viewer = None
        # Set while a render is in flight; both are None when idle.
        self.worker = None
        self.worker_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Mosaic Generator")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Guide Image Section
        guide_image_group = QGroupBox("Guide Image")
        guide_image_layout = QVBoxLayout()

        # Button to select guide image
        select_image_btn = QPushButton("Select Guide Image")
        select_image_btn.clicked.connect(self.select_guide_image)
        guide_image_layout.addWidget(select_image_btn)

        # Label to display the guide image
        self.guide_image_label = QLabel("No image selected")
        self.guide_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.guide_image_label.setFixedSize(300, 300)
        self.guide_image_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
            }
        """)
        self.guide_image_label.setScaledContents(False)
        guide_image_layout.addWidget(self.guide_image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        guide_image_group.setLayout(guide_image_layout)
        main_layout.addWidget(guide_image_group)

        # Tile Folder Section
        tile_folder_group = QGroupBox("Tile Images Folder")
        tile_folder_layout = QVBoxLayout()

        # Horizontal layout for path display and browse button
        path_layout = QHBoxLayout()

        # Line edit to display folder path
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("No folder selected")
        self.folder_path_edit.setReadOnly(True)
        path_layout.addWidget(self.folder_path_edit)

        # Browse button
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.select_tile_folder)
        browse_btn.setMaximumWidth(100)
        path_layout.addWidget(browse_btn)

        tile_folder_layout.addLayout(path_layout)

        # Checkbox for scanning subdirectories
        self.subdirs_checkbox = QCheckBox("Include subdirectories")
        self.subdirs_checkbox.setChecked(False)
        self.subdirs_checkbox.stateChanged.connect(self.update_scan_subdirectories)
        tile_folder_layout.addWidget(self.subdirs_checkbox)

        tile_folder_group.setLayout(tile_folder_layout)
        main_layout.addWidget(tile_folder_group)

        # Tile Settings Section
        tile_settings_group = QGroupBox("Tile Settings")
        tile_settings_layout = QVBoxLayout()

        tile_size_layout = QHBoxLayout()

        tile_size_layout.addWidget(QLabel("Tile width (px):"))
        self.tile_width_spinbox = QSpinBox()
        self.tile_width_spinbox.setRange(8, 2000)
        self.tile_width_spinbox.setValue(self.tile_width)
        self.tile_width_spinbox.valueChanged.connect(self.update_derived_dimensions)
        tile_size_layout.addWidget(self.tile_width_spinbox)

        tile_size_layout.addWidget(QLabel("Tile height (px):"))
        self.tile_height_spinbox = QSpinBox()
        self.tile_height_spinbox.setRange(8, 2000)
        self.tile_height_spinbox.setValue(self.tile_height)
        self.tile_height_spinbox.valueChanged.connect(self.update_derived_dimensions)
        tile_size_layout.addWidget(self.tile_height_spinbox)

        tile_size_layout.addStretch()
        tile_settings_layout.addLayout(tile_size_layout)

        # Aspect ratio readout - tiles are cropped to this ratio by the
        # preprocessor, so it is also the shape of every guide grid cell.
        self.tile_aspect_label = QLabel()
        self.tile_aspect_label.setEnabled(False)  # palette-aware secondary text
        tile_settings_layout.addWidget(self.tile_aspect_label)

        # Variety controls. Without these a plain nearest-neighbour match
        # reuses one photo across every flat area of the guide.
        variety_layout = QHBoxLayout()

        variety_layout.addWidget(QLabel("Max uses per image:"))
        self.max_reuse_spinbox = QSpinBox()
        self.max_reuse_spinbox.setRange(0, 9999)
        self.max_reuse_spinbox.setValue(0)
        self.max_reuse_spinbox.setSpecialValueText("unlimited")
        self.max_reuse_spinbox.setToolTip(
            "Cap how many times one image may appear. 0 means no limit.\n"
            "Best effort: if no alternative fits, the least-used image wins."
        )
        variety_layout.addWidget(self.max_reuse_spinbox)

        variety_layout.addWidget(QLabel("Min gap between repeats:"))
        self.min_distance_spinbox = QSpinBox()
        self.min_distance_spinbox.setRange(0, 50)
        self.min_distance_spinbox.setValue(0)
        self.min_distance_spinbox.setSpecialValueText("none")
        self.min_distance_spinbox.setToolTip(
            "Keep repeats of the same image at least this many cells apart."
        )
        variety_layout.addWidget(self.min_distance_spinbox)

        variety_layout.addStretch()
        tile_settings_layout.addLayout(variety_layout)

        # Soft variety and colour tint. Unlike the hard limits above, these
        # trade colour accuracy for variety gradually rather than all at once.
        blend_layout = QHBoxLayout()

        blend_layout.addWidget(QLabel("Variety:"))
        self.variety_spinbox = QSpinBox()
        self.variety_spinbox.setRange(0, 100)
        self.variety_spinbox.setValue(0)
        self.variety_spinbox.setSpecialValueText("off")
        self.variety_spinbox.setToolTip(
            "Favour images that have been used less, 0-100.\n"
            "Each use makes an image slightly less likely to be picked again,\n"
            "so more of your library appears at some cost to colour accuracy."
        )
        blend_layout.addWidget(self.variety_spinbox)

        blend_layout.addWidget(QLabel("Colour tint:"))
        self.tint_spinbox = QSpinBox()
        self.tint_spinbox.setRange(0, 100)
        self.tint_spinbox.setValue(0)
        self.tint_spinbox.setSuffix("%")
        self.tint_spinbox.setSpecialValueText("off")
        self.tint_spinbox.setToolTip(
            "Shift each image's colours toward the part of the guide it covers.\n"
            "Makes loosely matched images read correctly, so Variety can be\n"
            "raised without the mosaic going muddy. 15-30% is usually subtle."
        )
        blend_layout.addWidget(self.tint_spinbox)

        blend_layout.addStretch()
        tile_settings_layout.addLayout(blend_layout)

        self.randomize_order_checkbox = QCheckBox("Randomize placement order")
        self.randomize_order_checkbox.setChecked(True)
        self.randomize_order_checkbox.setToolTip(
            "Fill cells in a shuffled order rather than row by row. When\n"
            "Variety or a limit is on, this spreads the compromise evenly\n"
            "instead of leaving the bottom rows with the leftover images."
        )
        tile_settings_layout.addWidget(self.randomize_order_checkbox)

        tile_settings_group.setLayout(tile_settings_layout)
        main_layout.addWidget(tile_settings_group)

        # Output Settings Section
        output_settings_group = QGroupBox("Output Settings")
        output_settings_layout = QVBoxLayout()

        # Width and height spin boxes
        size_layout = QHBoxLayout()

        size_layout.addWidget(QLabel("Width (inches):"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 100)
        self.width_spinbox.setValue(self.output_width_inches)
        self.width_spinbox.valueChanged.connect(self.update_derived_dimensions)
        size_layout.addWidget(self.width_spinbox)

        size_layout.addWidget(QLabel("Height (inches):"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 100)
        self.height_spinbox.setValue(self.output_height_inches)
        self.height_spinbox.valueChanged.connect(self.update_derived_dimensions)
        size_layout.addWidget(self.height_spinbox)

        size_layout.addStretch()
        output_settings_layout.addLayout(size_layout)

        # Pixel dimensions label
        self.output_dimensions_label = QLabel()
        output_settings_layout.addWidget(self.output_dimensions_label)

        # Derived tile grid readout
        self.grid_info_label = QLabel()
        self.grid_info_label.setEnabled(False)  # palette-aware secondary text
        output_settings_layout.addWidget(self.grid_info_label)

        # All labels exist now, so it is safe to populate them.
        self.update_derived_dimensions()

        # Where the finished mosaic is written
        output_path_layout = QHBoxLayout()
        output_path_layout.addWidget(QLabel("Save to:"))

        self.output_folder_edit = QLineEdit(self.output_folder)
        self.output_folder_edit.setReadOnly(True)
        output_path_layout.addWidget(self.output_folder_edit)

        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.select_output_folder)
        output_browse_btn.setMaximumWidth(100)
        output_path_layout.addWidget(output_browse_btn)

        self.output_format_combo = QComboBox()
        for label, ext in OUTPUT_FORMATS:
            self.output_format_combo.addItem(label, ext)
        output_path_layout.addWidget(self.output_format_combo)

        output_settings_layout.addLayout(output_path_layout)

        output_settings_group.setLayout(output_settings_layout)
        main_layout.addWidget(output_settings_group)

        # Generate Mosaic Section
        generate_layout = QHBoxLayout()
        generate_layout.addStretch()

        self.generate_btn = QPushButton("Generate Mosaic")
        self.generate_btn.setEnabled(False)
        self.generate_btn.setMinimumWidth(150)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_mosaic)
        generate_layout.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumWidth(100)
        self.cancel_btn.setStyleSheet("QPushButton { padding: 10px; }")
        self.cancel_btn.clicked.connect(self.cancel_mosaic)
        self.cancel_btn.setVisible(False)
        generate_layout.addWidget(self.cancel_btn)

        generate_layout.addStretch()

        main_layout.addLayout(generate_layout)

        # Progress Section
        self.progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        # Status label showing current task
        self.progress_status_label = QLabel("Ready")
        self.progress_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                padding: 5px;
            }
        """)
        progress_layout.addWidget(self.progress_status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        self.progress_group.setLayout(progress_layout)
        self.progress_group.setVisible(False)  # Hidden by default
        main_layout.addWidget(self.progress_group)

        # Add stretch to push everything to the top
        main_layout.addStretch()

    def select_guide_image(self):
        """Open file dialog to select guide image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Guide Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff "
            "*.heic *.heif *.dng);;All Files (*)"
        )

        if file_path:
            self.guide_image_path = file_path
            self.display_guide_image(file_path)
            self.check_ready_to_generate()

    def display_guide_image(self, image_path):
        """Display the selected guide image in the label"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            max_dim = 400
            scaled_pixmap = pixmap.scaled(
                max_dim, max_dim,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.guide_image_label.setFixedSize(scaled_pixmap.size())
            self.guide_image_label.setPixmap(scaled_pixmap)
        else:
            self.guide_image_label.setText("Failed to load image")

    def update_derived_dimensions(self):
        """Recompute output pixel size and the tile grid from the spin boxes.

        Single entry point for every size control, so the pixel readout, the
        tile aspect ratio and the grid summary can never drift apart.
        """
        self.output_width_inches = self.width_spinbox.value()
        self.output_height_inches = self.height_spinbox.value()
        self.tile_width = self.tile_width_spinbox.value()
        self.tile_height = self.tile_height_spinbox.value()

        w = self.output_width_px
        h = self.output_height_px
        self.output_dimensions_label.setText(
            f"Output: {w} \u00d7 {h} px ({self.output_dpi} DPI)"
        )

        self.tile_aspect_label.setText(
            f"Tile aspect ratio: {self.tile_aspect_ratio:.3f} "
            f"({self.tile_width}:{self.tile_height})"
        )

        cols, rows = self.grid_cols, self.grid_rows
        if cols == 0 or rows == 0:
            self.grid_info_label.setText(
                "\u26a0 Tile is larger than the output canvas - no tiles fit."
            )
            self.grid_info_label.setEnabled(True)
            self.grid_info_label.setStyleSheet("QLabel { color: #e74c3c; }")
            return

        rem_x, rem_y = self.remainder_px
        text = (f"Grid: {cols} \u00d7 {rows} = {self.total_tiles:,} tiles")
        if rem_x or rem_y:
            text += f"  \u00b7  {rem_x} \u00d7 {rem_y} px unused at edges"
        self.grid_info_label.setText(text)
        self.grid_info_label.setStyleSheet("")
        self.grid_info_label.setEnabled(False)

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

    def select_tile_folder(self):
        """Open folder dialog to select tile images folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Tile Images Folder",
            ""
        )

        if folder_path:
            self.tile_folder_path = folder_path
            self.folder_path_edit.setText(folder_path)
            self.check_ready_to_generate()

    def select_output_folder(self):
        """Open folder dialog to choose where mosaics are saved"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
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

    def update_scan_subdirectories(self, state):
        """Update the scan subdirectories setting based on checkbox state"""
        self.scan_subdirectories = state == Qt.CheckState.Checked.value

    def check_ready_to_generate(self):
        """Enable generate button if both guide image and tile folder are selected"""
        if self.guide_image_path and self.tile_folder_path:
            self.generate_btn.setEnabled(True)
        else:
            self.generate_btn.setEnabled(False)

    # Progress UI Methods

    def show_progress(self):
        """Show the progress section and prepare for a new task."""
        self.progress_group.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_status_label.setText("Starting...")
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setVisible(True)

    def hide_progress(self):
        """Hide the progress section and reset to ready state."""
        self.progress_group.setVisible(False)
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
            variety=self.variety_spinbox.value(),
            randomize_order=self.randomize_order_checkbox.isChecked(),
            tint_strength=self.tint_spinbox.value(),
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
        stem = "mosaic"
        if self.guide_image_path:
            stem = os.path.splitext(os.path.basename(self.guide_image_path))[0]
            stem = f"{stem}_mosaic"
        ext = self.output_format_combo.currentData()
        return unique_path(self.output_folder, stem, ext)

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

    def show_result(self, output_path: str, summary: str):
        """Open the finished mosaic in its own window for inspection.

        Modeless, so the main window stays usable while it is open.
        """
        self.result_viewer = ResultViewer(output_path, summary, self)
        self.result_viewer.show()

    def on_render_failed(self, message: str):
        self._teardown_worker()
        self.hide_progress()
        QMessageBox.critical(self, "Mosaic failed", message)

    def on_render_cancelled(self):
        self._teardown_worker()
        self.hide_progress()
        self.set_progress_status("Ready")

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
