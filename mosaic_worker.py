"""
Mosaic Worker Module

Runs the full mosaic pipeline off the GUI thread.

A mosaic takes minutes, not milliseconds: analysing a large tile library is
the dominant cost, and doing any of it on the GUI thread makes the window
stop responding. MosaicWorker is a QObject designed to be moved onto a
QThread; it reports progress by signal and never touches a widget.
"""

import logging
import os
from dataclasses import dataclass, field
from threading import Event
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from guide_image import GuideImage
from mosaic_renderer import (
    MosaicRenderer,
    RenderCancelled,
    RenderConfig,
    RenderStats,
)
from tile_database import TileDatabase
from tile_preprocessor import PreprocessorConfig, TilePreprocessor

logger = logging.getLogger(__name__)


class CancellationToken:
    """A thread-safe cancel flag that is deliberately not a QObject.

    Cancelling has to take effect while the worker thread is busy inside
    run(), so the flag must be settable from another thread immediately.
    A QObject slot cannot do that: Qt delivers a cross-thread invocation
    through the receiver's event loop, and the worker's event loop does not
    run again until the job it is executing has already finished. A plain
    object has no thread affinity, so calling cancel() on it always runs in
    the caller's thread and sets the flag at once.
    """

    def __init__(self):
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


@dataclass
class MosaicJob:
    """An immutable snapshot of everything a render needs.

    Taken on the GUI thread before the worker starts, so the worker never
    reads a widget while the user is still editing one.
    """

    guide_image_path: str
    tile_folder_path: str
    output_path: str

    tile_width: int = 100
    tile_height: int = 100
    output_width_px: int = 6000
    output_height_px: int = 9000
    output_dpi: int = 300

    scan_subdirectories: bool = False
    max_tile_reuse: int = 0
    min_reuse_distance: int = 0
    variety: int = 0
    randomize_order: bool = True
    tint_strength: int = 0
    cache_dir: Optional[str] = None
    #: Worker processes for tile analysis; None picks a capped core count.
    max_workers: Optional[int] = None

    def preprocessor_config(self) -> PreprocessorConfig:
        """Tile dimensions come from one place so cells and tiles agree."""
        return PreprocessorConfig(
            target_width=self.tile_width,
            target_height=self.tile_height,
            cache_dir=self.cache_dir,
        )

    def render_config(self) -> RenderConfig:
        return RenderConfig(
            tile_width=self.tile_width,
            tile_height=self.tile_height,
            max_tile_reuse=self.max_tile_reuse,
            min_reuse_distance=self.min_reuse_distance,
            variety=self.variety,
            randomize_order=self.randomize_order,
            tint_strength=self.tint_strength,
        )


# Relative cost of each stage, used to turn per-stage progress into one
# overall percentage. Tile analysis dominates; the rest is comparatively
# cheap. These are rough but keep the bar from stalling then leaping.
STAGE_WEIGHTS = {
    "Loading tiles...": 0.60,
    "Analysing guide image...": 0.10,
    "Matching tiles...": 0.15,
    "Assembling mosaic...": 0.12,
    "Saving...": 0.03,
}
STAGE_ORDER = list(STAGE_WEIGHTS)


class MosaicWorker(QObject):
    """Runs a MosaicJob and reports progress by signal.

    Move onto a QThread and connect `run` to the thread's `started` signal.
    Exactly one of finished/cancelled/failed is emitted per run.
    """

    #: (stage name, completed, total) for the current stage
    stage_progress = pyqtSignal(str, int, int)
    #: Overall completion, 0-100, across all stages
    overall_progress = pyqtSignal(int)
    #: Output path and a summary of the plan
    finished = pyqtSignal(str, object)
    #: Human-readable failure message
    failed = pyqtSignal(str)
    #: Emitted instead of finished when the user cancels
    cancelled = pyqtSignal()

    def __init__(self, job: MosaicJob,
                 token: Optional[CancellationToken] = None,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self.job = job
        self.token = token or CancellationToken()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation. Safe to call from the GUI thread.

        Call this as a plain method. Do NOT wire it up as the receiver of a
        Qt signal or QTimer (`someSignal.connect(worker.cancel)`): because
        the worker lives in another thread, Qt would queue the call onto
        that thread's event loop, which is blocked running the job, so the
        cancel would not arrive until the render had already finished.
        Connect a GUI-thread slot that calls this instead, or hand out
        `worker.token`, which has no thread affinity.
        """
        self.token.cancel()

    def is_cancelled(self) -> bool:
        return self.token.is_cancelled()

    # ------------------------------------------------------------------
    # Progress plumbing
    # ------------------------------------------------------------------

    def _report(self, stage: str, completed: int, total: int) -> None:
        """Emit one stage's progress and the derived overall percentage."""
        self.stage_progress.emit(stage, completed, total)

        done_before = sum(STAGE_WEIGHTS[s]
                          for s in STAGE_ORDER[:STAGE_ORDER.index(stage)])
        fraction = (completed / total) if total else 1.0
        overall = (done_before + STAGE_WEIGHTS[stage] * fraction) * 100
        self.overall_progress.emit(max(0, min(100, int(overall))))

    def _callback(self, stage: str):
        """Adapt a stage name to the (completed, total, message) callback."""
        def report(completed, total, _message):
            self._report(stage, completed, total)
        return report

    # ------------------------------------------------------------------
    # The pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the job. Emits exactly one terminal signal."""
        try:
            stats = self._run_pipeline()
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as e:                      # noqa: BLE001
            logger.exception("Mosaic generation failed")
            self.failed.emit(str(e))
        else:
            if stats is None:
                self.cancelled.emit()
            else:
                self.finished.emit(self.job.output_path, stats)

    def _run_pipeline(self) -> Optional[RenderStats]:
        job = self.job

        preprocessor = TilePreprocessor(job.preprocessor_config())

        # --- Load and analyse the tile library (the slow part) ----------
        self._report("Loading tiles...", 0, 1)
        database = TileDatabase(preprocessor=preprocessor)
        loaded = database.load_tiles_parallel(
            job.tile_folder_path,
            job.preprocessor_config(),
            recursive=job.scan_subdirectories,
            max_workers=job.max_workers,
            progress_callback=self._callback("Loading tiles..."),
            should_cancel=self.is_cancelled,
        )

        if self.is_cancelled():
            return None
        if loaded == 0:
            raise RuntimeError(
                f"No usable tile images found in {job.tile_folder_path}"
                + (" or its subfolders." if job.scan_subdirectories else ".")
            )

        # --- Analyse the guide image ------------------------------------
        self._report("Analysing guide image...", 0, 1)
        guide = GuideImage(
            job.guide_image_path,
            job.output_width_px,
            job.output_height_px,
            job.tile_width,
            job.tile_height,
        )
        self._report("Analysing guide image...", 1, 1)

        if self.is_cancelled():
            return None

        # --- Match and assemble ------------------------------------------
        renderer = MosaicRenderer(database, job.render_config(),
                                  preprocessor=preprocessor)

        placements = renderer.plan(
            guide,
            progress_callback=self._callback("Matching tiles..."),
            should_cancel=self.is_cancelled,
        )
        mosaic = renderer.composite(
            placements,
            guide.grid_dimensions,
            progress_callback=self._callback("Assembling mosaic..."),
            should_cancel=self.is_cancelled,
        )

        # --- Save ---------------------------------------------------------
        self._report("Saving...", 0, 1)
        self._save(mosaic)
        self._report("Saving...", 1, 1)

        return renderer.summarize(placements)

    def _save(self, mosaic) -> None:
        """Write the mosaic, stamping the DPI the output was sized for."""
        path = self.job.output_path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        save_args = {"dpi": (self.job.output_dpi, self.job.output_dpi)}
        if os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg'):
            # A mosaic is full of hard tile edges, which low-quality JPEG
            # turns into visible ringing.
            save_args.update(quality=95, subsampling=0)
            mosaic = mosaic.convert('RGB')

        mosaic.save(path, **save_args)
