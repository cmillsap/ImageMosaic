# Image Mosaic Generator

A desktop application built with PyQt6 that allows users to generate image mosaics from a base image and a collection of tile images.

## Features

- Select a base image that will be the foundation of your mosaic
- Preview the selected base image in the UI
- Browse and select a folder containing tile images
- Clean, intuitive user interface

## Installation

1. Make sure you have Python 3.8 or higher installed

2. Create and activate a virtual environment:

**On Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your virtual environment is activated (you should see `(venv)` in your terminal prompt)

2. Run the application:
```bash
python mosaic_app.py
```

3. To deactivate the virtual environment when done:
```bash
deactivate
```

### Using the Application

1. Click "Select Base Image" to choose the image you want to convert into a mosaic
2. The selected image will be displayed in the preview window
3. Click "Browse" to select a folder containing the images that will be used as mosaic tiles
4. The folder path will be displayed in the text field
5. Once both a base image and tile folder are selected, the "Generate Mosaic" button will be enabled (functionality coming soon)

## Project Structure

```
ImageMosaic/
├── venv/                # Virtual environment (not tracked in git)
├── mosaic_app.py        # Main application file
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.8+
- PyQt6
- Pillow (PIL)

## Future Enhancements

- Implement mosaic generation algorithm
- Add customization options (tile size, color matching, etc.)
- Save generated mosaics
- Progress bar for mosaic generation
