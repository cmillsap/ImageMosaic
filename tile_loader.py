"""
Parallel Tile Loader

Analysing a tile library is the slowest part of building a mosaic - roughly
80 ms per photo even after decoding and averaging are optimised, so a
10,000-image library costs about ten minutes on one core. The work is
independent per file, so it spreads across processes cleanly.

Processes rather than threads: the work is CPU-bound Python (PIL decode,
NumPy averaging, Haar detection), and the GIL would serialise most of it.

This module deliberately imports no PyQt: on Windows every worker is a
fresh interpreter that re-imports it, and pulling in a GUI toolkit per
child would cost more than it saves.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional, Sequence

from tile_analyzer import ImageTileAnalyzer, TileData
from tile_preprocessor import PreprocessorConfig, TilePreprocessor

logger = logging.getLogger(__name__)

# Measured on a 16-core machine over a real 8,000-photo library: 8 and 16
# workers both reached ~3.1x, so the smaller pool is the default - same
# throughput for half the processes and half the memory. The ceiling is
# well under the core count because OpenCV already threads Haar detection
# internally, so the single-process baseline is itself partly parallel.
DEFAULT_MAX_WORKERS = 8

# Files handed to each worker per batch. Large enough that per-task
# overhead disappears, small enough that progress still moves visibly.
CHUNK_SIZE = 16

# Below this many files, staying serial is faster than starting a pool.
# Spinning up 8 workers costs ~0.8 s (Windows spawns fresh interpreters
# that each re-import OpenCV and NumPy), against a saving of ~55 ms per
# file, so the pool only pays for itself past roughly 15 files.
PARALLEL_THRESHOLD = 32

_state = None


def _init_worker(config: PreprocessorConfig) -> None:
    """Build one preprocessor per worker process.

    The detectors hold OpenCV cascade objects, which cannot be pickled, so
    each process constructs its own rather than receiving one.
    """
    global _state
    _state = (TilePreprocessor(config), ImageTileAnalyzer())


def _analyse_one(path: str):
    """Analyse a single tile. Returns TileData, or None if unreadable.

    Real libraries contain truncated and corrupt files, and one bad file
    must not abort a ten-minute run, so failures come back as None.
    """
    preprocessor, analyzer = _state
    try:
        image = preprocessor.preprocess_image(path)
        return analyzer.analyze_pil_image(image, path)
    except Exception as e:                                  # noqa: BLE001
        logger.debug("Skipping unreadable tile %s: %s", path, e)
        return None


def resolve_worker_count(requested: Optional[int] = None) -> int:
    """Pick a worker count: explicit value, else a capped core count."""
    if requested is not None and requested > 0:
        return requested
    return max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))


def analyse_paths(paths: Sequence[str],
                  config: PreprocessorConfig,
                  max_workers: Optional[int] = None,
                  progress_callback: Optional[Callable[[int, int, str], None]] = None,
                  should_cancel: Optional[Callable[[], bool]] = None,
                  ) -> List[TileData]:
    """
    Analyse tile images across a process pool.

    Args:
        paths: Image files to analyse
        config: Preprocessor settings, rebuilt inside each worker
        max_workers: Process count; defaults to a capped core count
        progress_callback: Called as (completed, total, message)
        should_cancel: Polled as results arrive; stops early when True

    Returns:
        TileData for every file that could be read, in completion order.
        Unreadable files are skipped.

    Falls back to serial analysis if a process pool cannot be started.
    """
    paths = list(paths)
    total = len(paths)
    if total == 0:
        return []

    workers = resolve_worker_count(max_workers)
    if workers == 1 or total < PARALLEL_THRESHOLD:
        return _analyse_serial(paths, config, progress_callback, should_cancel)

    try:
        return _analyse_parallel(paths, config, workers,
                                 progress_callback, should_cancel)
    except Exception as e:                                  # noqa: BLE001
        # A frozen build, a sandbox with no process spawning, or a pool
        # that died. Slow beats broken.
        logger.warning("Parallel tile loading unavailable (%s); "
                       "falling back to serial.", e)
        return _analyse_serial(paths, config, progress_callback, should_cancel)


def _analyse_parallel(paths, config, workers, progress_callback, should_cancel):
    results: List[TileData] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(config,)) as executor:
        iterator = executor.map(_analyse_one, paths, chunksize=CHUNK_SIZE)
        for tile in iterator:
            completed += 1
            if tile is not None:
                results.append(tile)

            if should_cancel is not None and should_cancel():
                # cancel_futures drops work not yet started; tasks already
                # running still finish, which is why cancellation is not
                # instant with a pool.
                executor.shutdown(wait=False, cancel_futures=True)
                return results

            if progress_callback is not None and (
                    completed % CHUNK_SIZE == 0 or completed == len(paths)):
                progress_callback(completed, len(paths), "Loading tiles...")

    return results


def _analyse_serial(paths, config, progress_callback, should_cancel):
    _init_worker(config)
    results: List[TileData] = []

    for index, path in enumerate(paths):
        if should_cancel is not None and should_cancel():
            return results

        tile = _analyse_one(path)
        if tile is not None:
            results.append(tile)

        if progress_callback is not None:
            progress_callback(index + 1, len(paths), "Loading tiles...")

    return results
