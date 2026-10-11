"""
Parallel Tile Loader

Analysing a tile library is the slowest part of building a mosaic - roughly
80 ms per photo even after decoding and averaging are optimised, so a
10,000-image library costs about ten minutes on one core. The work is
independent per file, so it spreads across processes cleanly.

Processes rather than threads: the work is CPU-bound Python (PIL decode,
NumPy averaging, Haar detection), and the GIL would serialise most of it.

When the tile filter is on, each worker also prepares the tile for the
filter's model, but the model itself runs here in the calling process, in
batches: it needs about 350 MB once loaded, too much to copy into every
worker. See tile_filter.

This module deliberately imports no PyQt: on Windows every worker is a
fresh interpreter that re-imports it, and pulling in a GUI toolkit per
child would cost more than it saves.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import tile_filter
from image_io import open_image
from tile_analyzer import ImageTileAnalyzer, TileData
from tile_filter import FilterConfig, JunkScorer, ScoreCache
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

# Tiles scored by the filter's model per inference call.
SCORE_BATCH_SIZE = 32

# What a worker does for the tile filter, per file. CHECK runs only the
# cheap screen-capture test, for files whose model score is already
# cached; SCORE also prepares the tile for the model.
NO_FILTER, CHECK, SCORE = 0, 1, 2

_state = None


def _init_worker(config: PreprocessorConfig) -> None:
    """Build one preprocessor per worker process.

    The detectors hold OpenCV cascade objects, which cannot be pickled, so
    each process constructs its own rather than receiving one.
    """
    global _state
    _state = (TilePreprocessor(config), ImageTileAnalyzer())


def _analyse_one(path: str, filter_mode: int = NO_FILTER):
    """Analyse a single tile. Returns TileData, or None if unreadable.

    Real libraries contain truncated and corrupt files, and one bad file
    must not abort a ten-minute run, so failures come back as None.
    """
    preprocessor, analyzer = _state

    # The model sees the whole photo, not the cropped tile: a crop can cut
    # away the status bar or text that gives a screenshot away. Taking it
    # from the preprocessor's decode, before the crop, saves decoding every
    # photo twice - which made a first run over a library half as slow
    # again.
    model_input = []
    on_decoded = None
    if filter_mode == SCORE:
        def on_decoded(img):
            model_input.append(tile_filter.clip_input(img))

    try:
        image = preprocessor.preprocess_image(path, on_decoded=on_decoded)
        tile = analyzer.analyze_pil_image(image, path)
    except Exception as e:                                  # noqa: BLE001
        logger.debug("Skipping unreadable tile %s: %s", path, e)
        return None

    if filter_mode != NO_FILTER:
        _prepare_for_filter(tile, filter_mode,
                            model_input[0] if model_input else None)
    return tile


def _prepare_for_filter(tile: TileData, filter_mode: int,
                        model_input=None) -> None:
    """Run the filter's per-file test and, for SCORE, ready the model input.

    model_input is the tile as the model sees it, if the preprocessor's
    decode already produced it. When the tile came from the tile cache
    there was no decode, so the original is read here instead.
    """
    path = tile.image_path
    if tile_filter.is_screen_capture(path):
        tile.filter_result = (1.0, tile_filter.SCREEN_CAPTURE)
        return
    if filter_mode != SCORE:
        return
    if model_input is not None:
        tile.clip_input = model_input
        return
    try:
        size = 2 * tile_filter.CLIP_SIZE
        with open_image(path, draft_size=(size, size)) as img:
            tile.clip_input = tile_filter.clip_input(img)
    except Exception as e:                                  # noqa: BLE001
        # The tile itself was readable, so keep it, unfiltered.
        logger.debug("Tile filter could not read %s: %s", path, e)


class FilterUnavailable(RuntimeError):
    """The tile filter's model could not be loaded."""


class _FilterScoring:
    """The calling process's half of the tile filter.

    Looks up cached scores before the workers start, batches the tiles that
    need the model, and fills in each tile's filter_result.
    """

    def __init__(self, paths: List[str], config: FilterConfig):
        self._cache = ScoreCache(config.cache_dir)
        self._cached = self._cache.get_many(paths)
        self.modes = [CHECK if p in self._cached else SCORE for p in paths]
        self._pending: List[TileData] = []
        self._new_scores: Dict[str, Tuple[float, str]] = {}

        # Loaded now, before any worker starts, so a broken model fails at
        # once rather than after the pool has worked through the library.
        self._scorer = None
        if SCORE in self.modes:
            try:
                self._scorer = JunkScorer(config.model_path)
            except Exception as e:                          # noqa: BLE001
                self._cache.close()
                raise FilterUnavailable(
                    "The screenshot filter could not start. Turn off "
                    "\"Skip screenshots, maps & documents\" to make the "
                    f"mosaic without it.\n\n{e}") from e

    def add(self, tile: TileData) -> None:
        if tile.clip_input is not None:
            self._pending.append(tile)
            if len(self._pending) >= SCORE_BATCH_SIZE:
                self.flush()
        elif tile.filter_result is None:
            tile.filter_result = self._cached.get(tile.image_path)

    def flush(self) -> None:
        if not self._pending:
            return
        results = self._scorer.score([t.clip_input for t in self._pending])
        for tile, result in zip(self._pending, results):
            tile.filter_result = result
            tile.clip_input = None
            self._new_scores[tile.image_path] = result
        self._pending.clear()

    def finish(self) -> None:
        """Score what is left and save new scores for next time."""
        try:
            self.flush()
            self._cache.put_many(self._new_scores)
        finally:
            self._cache.close()

    def abandon(self) -> None:
        """Stop early, still keeping the scores already paid for."""
        self._pending.clear()
        self._cache.put_many(self._new_scores)
        self._cache.close()


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
                  filter_config: Optional[FilterConfig] = None,
                  ) -> List[TileData]:
    """
    Analyse tile images across a process pool.

    Args:
        paths: Image files to analyse
        config: Preprocessor settings, rebuilt inside each worker
        max_workers: Process count; defaults to a capped core count
        progress_callback: Called as (completed, total, message)
        should_cancel: Polled as results arrive; stops early when True
        filter_config: Run the tile filter too, setting each tile's
            filter_result. Nothing is excluded here; the caller decides.

    Returns:
        TileData for every file that could be read, in completion order.
        Unreadable files are skipped.

    Raises:
        FilterUnavailable: If the tile filter's model cannot be loaded

    Falls back to serial analysis if a process pool cannot be started.
    """
    paths = list(paths)
    total = len(paths)
    if total == 0:
        return []

    def serial(**filtering):
        return _analyse_serial(paths, config, progress_callback,
                               should_cancel, **filtering)

    def parallel(**filtering):
        return _analyse_parallel(paths, config, workers, progress_callback,
                                 should_cancel, **filtering)

    workers = resolve_worker_count(max_workers)
    if workers == 1 or total < PARALLEL_THRESHOLD:
        return _analyse_filtered(serial, paths, should_cancel, filter_config)

    try:
        return _analyse_filtered(parallel, paths, should_cancel,
                                 filter_config)
    except FilterUnavailable:
        # Not a pool problem: going serial would only fail the same way.
        raise
    except Exception as e:                                  # noqa: BLE001
        # A frozen build, a sandbox with no process spawning, or a pool
        # that died. Slow beats broken.
        logger.warning("Parallel tile loading unavailable (%s); "
                       "falling back to serial.", e)
        return _analyse_filtered(serial, paths, should_cancel, filter_config)


def _analyse_filtered(analyse, paths, should_cancel, filter_config):
    """Call analyse(), wiring in the tile filter when one is configured."""
    if filter_config is None:
        return analyse()

    scoring = _FilterScoring(paths, filter_config)
    try:
        results = analyse(modes=scoring.modes, on_tile=scoring.add)
    except BaseException:
        scoring.abandon()
        raise
    if should_cancel is not None and should_cancel():
        scoring.abandon()
    else:
        scoring.finish()
    return results


def _analyse_parallel(paths, config, workers, progress_callback, should_cancel,
                      modes=None, on_tile=None):
    results: List[TileData] = []
    completed = 0
    modes = modes or [NO_FILTER] * len(paths)

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(config,)) as executor:
        iterator = executor.map(_analyse_one, paths, modes,
                                chunksize=CHUNK_SIZE)
        for tile in iterator:
            completed += 1
            if tile is not None:
                if on_tile is not None:
                    on_tile(tile)
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


def _analyse_serial(paths, config, progress_callback, should_cancel,
                    modes=None, on_tile=None):
    _init_worker(config)
    results: List[TileData] = []
    modes = modes or [NO_FILTER] * len(paths)

    for index, path in enumerate(paths):
        if should_cancel is not None and should_cancel():
            return results

        tile = _analyse_one(path, modes[index])
        if tile is not None:
            if on_tile is not None:
                on_tile(tile)
            results.append(tile)

        if progress_callback is not None:
            progress_callback(index + 1, len(paths), "Loading tiles...")

    return results
