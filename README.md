# Image Mosaic Generator

A desktop application built with PyQt6 that rebuilds a guide image out of a
library of tile photographs, matching each cell of the image to the closest
photo by colour.

## How it works

The guide image is divided into a grid of cells. Every cell and every tile
is reduced to nine average colours (a 3x3 grid over the image), giving a
27-dimensional colour signature. Each cell is then matched to its nearest
tile using a k-nearest-neighbour index, and the chosen tiles are composited
into the finished mosaic.

Tiles are cropped to a uniform aspect ratio before being indexed. Cropping
is subject-aware: OpenCV looks for faces and, failing that, for the most
salient region, so a portrait is cropped around the face rather than
through it.

| Module | Responsibility |
| --- | --- |
| `tile_analyzer.py` | Splits an image into 9 sections and averages each |
| `tile_preprocessor.py` | Face/saliency-aware cropping to a uniform ratio |
| `tile_database.py` | k-NN colour index over the tile library |
| `guide_image.py` | Divides the guide image into analysed grid cells |
| `mosaic_renderer.py` | Chooses a tile per cell, then composites |
| `mosaic_worker.py` | Runs the pipeline off the GUI thread |
| `mosaic_app.py` | PyQt6 user interface |

## Installation

Requires Python 3.8 or higher.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

`requirements.txt` pins **opencv-contrib-python**, not `opencv-python`. The
saliency detector lives in the contrib package; with the base package it
silently disables itself and every tile falls back to a centre crop.

## Usage

```bash
python mosaic_app.py
```

1. **Select Guide Image** - the picture the mosaic will reproduce.
2. **Browse** to the folder of tile images, ticking *Include subdirectories*
   to scan nested folders.
3. Set the **tile size** in pixels. This fixes both the shape each tile is
   cropped to and the size of each guide grid cell, and the panel shows the
   resulting grid, e.g. `Grid: 60 x 90 = 5,400 tiles`.
4. Optionally limit repetition (see below).
5. Set the **output size** in inches; output is rendered at 300 DPI.
6. **Generate Mosaic**, choose where to save, and watch the progress bar.
   Generation runs on a background thread and can be cancelled at any time.

### Controlling repetition

A plain nearest-neighbour match reuses one photo across any large flat area
of the guide - a clear sky becomes the same image hundreds of times. Two
optional limits counter this:

- **Max uses per image** caps how often one photo may appear.
- **Min gap between repeats** keeps copies of a photo at least that many
  cells apart.

Both are best effort. When no candidate satisfies them, the least-used
candidate is chosen rather than the closest, so a render always completes.
Tighter limits mean more variety but looser colour matching, and a cap that
is arithmetically impossible (fewer tiles x uses than cells) is reported in
the log.

## Performance

Analysing the tile library dominates runtime - roughly 70 ms per tile, so a
10,000-image library takes about 11 minutes on first run. Preprocessed tiles
are cached on disk between runs, so repeat renders over the same library are
substantially faster.

## Tests

```bash
python -m pytest
```

Some detection tests need real photographs, since Haar cascades do not
respond to synthetic shapes. Place `face.jpg` (containing a face) and
`noface.tif` (containing none) in the project root to enable them; they are
gitignored and skip cleanly when absent.

## Requirements

- Python 3.8+
- PyQt6, Pillow, NumPy, scikit-learn, opencv-contrib-python

## Possible enhancements

- Expose the candidate pool size, which trades colour accuracy for variety
- Preview the mosaic before saving
- Alternative colour metrics (perceptual rather than RGB distance)
