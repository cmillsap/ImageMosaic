import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QLineEdit,
                              QFileDialog, QGroupBox, QSizePolicy, QSpinBox,
                              QCheckBox, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap

from tile_preprocessor import PreprocessorConfig


class MosaicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.guide_image_path = None
        self.tile_folder_path = None
        self.scan_subdirectories = False
        self.output_dpi = 300
        self.output_width_inches = 20
        self.output_height_inches = 30
        self.tile_width = 100
        self.tile_height = 100
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
        self.tile_aspect_label.setStyleSheet("QLabel { color: #555; }")
        tile_settings_layout.addWidget(self.tile_aspect_label)

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
        self.grid_info_label.setStyleSheet("QLabel { color: #555; }")
        output_settings_layout.addWidget(self.grid_info_label)

        # All labels exist now, so it is safe to populate them.
        self.update_derived_dimensions()

        output_settings_group.setLayout(output_settings_layout)
        main_layout.addWidget(output_settings_group)

        # Generate Mosaic Section (placeholder for future functionality)
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
        self.generate_btn.clicked.connect(self.demo_generate_mosaic)
        generate_layout.addWidget(self.generate_btn)
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
                color: #555;
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
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
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
            self.grid_info_label.setStyleSheet("QLabel { color: #c0392b; }")
            return

        rem_x, rem_y = self.remainder_px
        text = (f"Grid: {cols} \u00d7 {rows} = {self.total_tiles:,} tiles")
        if rem_x or rem_y:
            text += f"  \u00b7  {rem_x} \u00d7 {rem_y} px unused at edges"
        self.grid_info_label.setText(text)
        self.grid_info_label.setStyleSheet("QLabel { color: #555; }")

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

    def hide_progress(self):
        """Hide the progress section and reset to ready state."""
        self.progress_group.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_status_label.setText("Ready")
        self.check_ready_to_generate()

    def set_progress_status(self, status: str):
        """
        Set the current task status text.

        Args:
            status: Description of the current task (e.g., "Loading tiles...",
                   "Analyzing guide image...", "Generating mosaic...")
        """
        self.progress_status_label.setText(status)
        # Process events to update UI immediately
        QApplication.processEvents()

    def set_progress_value(self, value: int):
        """
        Set the progress bar value.

        Args:
            value: Progress percentage (0-100)
        """
        self.progress_bar.setValue(max(0, min(100, value)))
        # Process events to update UI immediately
        QApplication.processEvents()

    def set_progress(self, status: str, value: int):
        """
        Set both status text and progress value at once.

        Args:
            status: Description of the current task
            value: Progress percentage (0-100)
        """
        self.progress_status_label.setText(status)
        self.progress_bar.setValue(max(0, min(100, value)))
        # Process events to update UI immediately
        QApplication.processEvents()

    def start_task(self, task_name: str):
        """
        Start a new subtask, resetting progress to 0.

        Args:
            task_name: Name of the task to display (e.g., "Loading tiles...")
        """
        self.progress_status_label.setText(task_name)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

    def complete_task(self):
        """Mark the current task as complete (sets progress to 100%)."""
        self.progress_bar.setValue(100)
        QApplication.processEvents()

    # Demo/Test Methods (remove when real generation is implemented)

    def demo_generate_mosaic(self):
        """
        Demo method to test the progress UI.
        Simulates the mosaic generation process with fake progress.
        Remove this method when real generation is implemented.
        """
        self.show_progress()

        # Define demo tasks with their simulated step counts
        self._demo_tasks = [
            ("Loading tiles...", 25),
            ("Preprocessing tiles...", 20),
            ("Analyzing guide image...", 10),
            ("Building tile database...", 15),
            ("Matching tiles to guide...", 40),
            ("Assembling mosaic...", 30),
            ("Saving output...", 5),
        ]
        self._demo_task_index = 0
        self._demo_step = 0
        self._demo_steps_for_task = 0

        # Start the demo with a timer
        self._demo_timer = QTimer()
        self._demo_timer.timeout.connect(self._demo_tick)
        self._start_next_demo_task()
        self._demo_timer.start(50)  # 50ms per tick for smooth animation

    def _start_next_demo_task(self):
        """Start the next demo task."""
        if self._demo_task_index < len(self._demo_tasks):
            task_name, steps = self._demo_tasks[self._demo_task_index]
            self._demo_steps_for_task = steps
            self._demo_step = 0
            self.start_task(task_name)
        else:
            # All tasks complete
            self._demo_timer.stop()
            self.set_progress("Complete!", 100)
            # Hide progress after a short delay
            QTimer.singleShot(1500, self.hide_progress)

    def _demo_tick(self):
        """Process one tick of the demo animation."""
        self._demo_step += 1
        progress = int((self._demo_step / self._demo_steps_for_task) * 100)
        self.set_progress_value(progress)

        if self._demo_step >= self._demo_steps_for_task:
            # Move to next task
            self._demo_task_index += 1
            self._start_next_demo_task()


def main():
    app = QApplication(sys.argv)
    window = MosaicApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
