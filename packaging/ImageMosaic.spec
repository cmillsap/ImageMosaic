# -*- mode: python ; coding: utf-8 -*-
r"""PyInstaller spec for the Image Mosaic Generator.

Builds a one-folder distribution (dist/ImageMosaic). One-folder rather than
one-file: a one-file build unpacks ~300 MB of Qt, OpenCV and SciPy to temp on
every launch, and the tile analyser's worker processes each pay that cost
again. The Inno Setup script wraps this folder into the installer.

    venv\Scripts\pyinstaller.exe packaging/ImageMosaic.spec --noconfirm
"""

import os

from PyInstaller.utils.hooks import collect_data_files

PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# cv2.data.haarcascades is read at runtime by FaceDetector, and the cv2 hook
# does not always carry the XML cascades across.
datas = collect_data_files("cv2", includes=["data/*.xml"])

a = Analysis(
    [os.path.join(PROJECT_ROOT, "mosaic_app.py")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # Imported through sklearn's lazy dispatch, so the analyser misses
        # them. sklearn.neighbors._typedefs is deliberately absent: it moved
        # to sklearn.utils._typedefs in 1.2 and listing it only produces a
        # spurious "hidden import not found" error at build time.
        "sklearn.neighbors._partition_nodes",
        "sklearn.utils._typedefs",
        "sklearn.utils._heap",
        "sklearn.utils._sorting",
        "sklearn.utils._vector_sentinel",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Qt's other bindings and the test stack would otherwise be dragged in by
    # optional imports inside PyQt6 and sklearn.
    excludes=[
        "PyQt5", "PySide2", "PySide6", "tkinter",
        "matplotlib", "pytest", "_pytest", "pandas", "IPython",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImageMosaic",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPECPATH, "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ImageMosaic",
)
