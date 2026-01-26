import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QLineEdit,
                              QFileDialog, QGroupBox, QSizePolicy, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap


class MosaicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.guide_image_path = None
        self.tile_folder_path = None
        self.output_dpi = 300
        self.output_width_inches = 20
        self.output_height_inches = 30
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
        tile_folder_group.setLayout(tile_folder_layout)
        main_layout.addWidget(tile_folder_group)

        # Output Settings Section
        output_settings_group = QGroupBox("Output Settings")
        output_settings_layout = QVBoxLayout()

        # Width and height spin boxes
        size_layout = QHBoxLayout()

        size_layout.addWidget(QLabel("Width (inches):"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 100)
        self.width_spinbox.setValue(self.output_width_inches)
        self.width_spinbox.valueChanged.connect(self.update_output_dimensions)
        size_layout.addWidget(self.width_spinbox)

        size_layout.addWidget(QLabel("Height (inches):"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 100)
        self.height_spinbox.setValue(self.output_height_inches)
        self.height_spinbox.valueChanged.connect(self.update_output_dimensions)
        size_layout.addWidget(self.height_spinbox)

        size_layout.addStretch()
        output_settings_layout.addLayout(size_layout)

        # Pixel dimensions label
        self.output_dimensions_label = QLabel()
        self.update_output_dimensions()
        output_settings_layout.addWidget(self.output_dimensions_label)

        output_settings_group.setLayout(output_settings_layout)
        main_layout.addWidget(output_settings_group)

        # Generate Mosaic Section (placeholder for future functionality)
        generate_layout = QHBoxLayout()
        generate_layout.addStretch()

        self.generate_btn = QPushButton("Generate Mosaic")
        self.generate_btn.setEnabled(False)  # Disabled for now
        self.generate_btn.setMinimumWidth(150)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        generate_layout.addWidget(self.generate_btn)
        generate_layout.addStretch()

        main_layout.addLayout(generate_layout)

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

    def update_output_dimensions(self):
        """Update the pixel dimensions label based on current spin box values"""
        self.output_width_inches = self.width_spinbox.value()
        self.output_height_inches = self.height_spinbox.value()
        w = self.output_width_px
        h = self.output_height_px
        self.output_dimensions_label.setText(
            f"Output: {w} \u00d7 {h} px ({self.output_dpi} DPI)"
        )

    @property
    def output_width_px(self):
        return self.output_width_inches * self.output_dpi

    @property
    def output_height_px(self):
        return self.output_height_inches * self.output_dpi

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

    def check_ready_to_generate(self):
        """Enable generate button if both guide image and tile folder are selected"""
        if self.guide_image_path and self.tile_folder_path:
            self.generate_btn.setEnabled(True)
        else:
            self.generate_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    window = MosaicApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
