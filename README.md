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
| `image_io.py` | Opens every format, including HEIC and DNG |
| `tile_analyzer.py` | Splits an image into 9 sections and averages each |
| `tile_preprocessor.py` | Face/saliency-aware cropping to a uniform ratio |
| `tile_database.py` | k-NN colour index over the tile library |
| `guide_image.py` | Divides the guide image into analysed grid cells |
| `tile_loader.py` | Parallel tile analysis across worker processes |
| `mosaic_renderer.py` | Chooses a tile per cell, then composites |
| `mosaic_worker.py` | Runs the pipeline off the GUI thread |
| `mosaic_app.py` | PyQt6 user interface |
| `result_viewer.py` | Zoomable window showing the finished mosaic |

## Installation

### Windows installer

Run `ImageMosaic-1.0.0-Setup.exe` and follow the prompts. Python is not
required - the installer carries its own. It installs per-user by default,
so there is no UAC prompt; choose *Install for all users* on the first page
to put it in `Program Files` instead. The app then appears in the Start
Menu, and uninstalls from Settings > Apps like anything else.

The installer is not code-signed, so SmartScreen shows "Windows protected
your PC" on first run. *More info* > *Run anyway* gets past it.

### From source

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
   Mosaics are saved to your Documents folder as `<guide name>_mosaic.png`;
   use **Browse** next to *Save to* to pick another folder (remembered next
   time), and the format box for JPEG or TIFF. An existing file is never
   overwritten - a repeat run is numbered instead.
6. **Generate Mosaic** and watch the progress bar. When it finishes, the
   mosaic opens in its own window: scroll to zoom, drag to pan, and close
   it when you are done.
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

Analysing the tile library dominates runtime; everything else is minor by
comparison. Measured on a real 8,229-photo library (28 GB, median 6.3 MP)
on a 16-core machine:

| | per photo | 8,229 photos |
| --- | --- | --- |
| Serial, no cache | 75 ms | 10.3 min |
| Parallel, cold cache | 28 ms | 3.9 min |
| Parallel, warm cache | 0.6 ms | 0.1 min |

A full 20x30in poster at 300 DPI (5,400 tiles) from a 1,922-photo folder
takes about 66 s cold: 52 s analysing tiles, 5 s on the guide, 1 s
matching, 1 s compositing and 7 s saving the 6000x9000 PNG.

Four things make that work:

- **JPEG draft decoding.** `Image.draft()` has libjpeg decode at 1/2, 1/4
  or 1/8 scale, so an 18 MP photo is never fully expanded just to be
  shrunk to 1000 px.
- **NumPy section averaging**, about 10x faster than summing pixel tuples.
- **Tiles cached at final size.** Cache entries are the finished tile, not
  the full-resolution crop, which is ~46x smaller (0.14 GB rather than
  5.3 GB for this library) and lets the renderer paste straight from cache
  - compositing drops from 32 s to under 1 s.
- **Parallel analysis** across worker processes, about 3x.

Two tempting changes that measurement ruled out: running face detection on
a smaller image loses a third of the faces even with `minSize` scaled
proportionally, and `cv2.setNumThreads(1)` makes a serial run 3x slower
because OpenCV already parallelises Haar detection internally.

### Formats

Every image, tile or guide, is opened through `image_io.open_image()`.
Pillow covers JPEG, PNG, TIFF, BMP, GIF and Canon `.cr2`; `pillow-heif`
adds iPhone HEIC/HEIF; and `.dng` is developed by `rawpy` (LibRaw), since
Pillow only returns its ~160 px embedded thumbnail. Other camera RAW
(`.crw`, `.cr3`, ...) is skipped: rawpy could read it, but it is untested.
Corrupt and truncated files are skipped with a warning rather than failing
the run.

These formats cost more than JPEG, which gets libjpeg's reduced-scale
decode for free. Measured per tile (serial, no cache):

| Format | per photo |
| --- | --- |
| JPEG, 14 MP | 120 ms |
| HEIC, 12 MP | 215 ms |
| DNG, 12 MP linear | 310 ms |

DNGs are developed at half size (skipping demosaicing), which is still
far above tile resolution. Most of their cost is LibRaw reading the ~20 MB
file, not the development itself.

## Building the installer

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Two stages, both driven by that script: PyInstaller freezes the app into
`dist\ImageMosaic`, then Inno Setup packs that folder into
`dist\ImageMosaic-1.0.0-Setup.exe`. PyInstaller is installed into the venv
on demand; Inno Setup 6 must already be present, via
`winget install JRSoftware.InnoSetup`. Pass `-SkipApp` to recompile only
the installer when the frozen app is already built.

| File | Purpose |
| --- | --- |
| `packaging/build.ps1` | Runs both stages |
| `packaging/ImageMosaic.spec` | PyInstaller configuration |
| `packaging/installer.iss` | Inno Setup configuration |
| `packaging/make_icon.py` | Regenerates `icon.ico`; committed, rarely needed |

The frozen build is ~354 MB, which solid LZMA2 compresses to an ~85 MB
setup. Three things in it are load-bearing:

- **`multiprocessing.freeze_support()` in `main()`.** Tile analysis spawns
  worker processes, and a frozen worker re-executes the exe. Without it
  each worker opens its own window instead of analysing tiles.
- **The Haar cascades are collected explicitly.** `FaceDetector` reads
  `cv2.data.haarcascades` at runtime and the stock OpenCV hook does not
  reliably carry those XML files across. Miss them and every crop quietly
  falls back to centre.
- **One folder, not one file.** A one-file build unpacks all 354 MB to temp
  on every launch, and each worker process pays that again.

To change the version, edit `AppVersion` in `packaging/installer.iss`.
Leave `AppId` alone - it is how Windows recognises an installed copy and
upgrades it in place rather than installing a second one.

## Tests

```bash
python -m pytest
```

Some detection tests need real photographs, since Haar cascades do not
respond to synthetic shapes. Place `face.jpg` (containing a face) and
`noface.tif` (containing none) in the project root to enable them. The
format tests likewise need a `test.HEIC` and a `test.DNG`. All four are
gitignored and skip cleanly when absent.

## Requirements

- Python 3.8+
- PyQt6, Pillow, pillow-heif, rawpy, NumPy, scikit-learn,
  opencv-contrib-python

## Possible enhancements

- Expose the candidate pool size, which trades colour accuracy for variety
- `rawpy` support for `.crw` / `.cr3` / full-resolution `.dng`
- Preview the mosaic before saving
- Alternative colour metrics (perceptual rather than RGB distance)
