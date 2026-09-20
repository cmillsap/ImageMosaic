"""
Tests for the Mosaic Worker Module.

The worker is exercised synchronously by calling run() on the test thread:
signals still fire through direct connections, and the result is
deterministic. One test drives a real QThread to prove the threaded path
and cancellation actually work.
"""

import os
import shutil
import tempfile

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication, QThread, QTimer

from mosaic_worker import (
    STAGE_ORDER,
    STAGE_WEIGHTS,
    CancellationToken,
    MosaicJob,
    MosaicWorker,
)


TILE = 10


@pytest.fixture(scope="session")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def tile_folder(temp_dir):
    folder = os.path.join(temp_dir, 'tiles')
    os.makedirs(folder)
    for i, color in enumerate([
        (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
        (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]):
        Image.new('RGB', (30, 30), color).save(
            os.path.join(folder, f'tile_{i}.png'))
    return folder


@pytest.fixture
def guide_file(temp_dir):
    img = Image.new('RGB', (60, 40), (255, 0, 0))
    for x in range(30, 60):
        for y in range(40):
            img.putpixel((x, y), (0, 0, 255))
    path = os.path.join(temp_dir, 'guide.png')
    img.save(path)
    return path


@pytest.fixture
def job(temp_dir, tile_folder, guide_file):
    return MosaicJob(
        guide_image_path=guide_file,
        tile_folder_path=tile_folder,
        output_path=os.path.join(temp_dir, 'out', 'mosaic.png'),
        tile_width=TILE, tile_height=TILE,
        output_width_px=60, output_height_px=40,
        cache_dir=os.path.join(temp_dir, 'cache'),
    )


def collect(worker):
    """Attach recorders to every signal and return them."""
    record = {'stages': [], 'overall': [], 'finished': [],
              'failed': [], 'cancelled': 0}
    worker.stage_progress.connect(
        lambda s, c, t: record['stages'].append((s, c, t)))
    worker.overall_progress.connect(record['overall'].append)
    worker.finished.connect(lambda p, s: record['finished'].append((p, s)))
    worker.failed.connect(record['failed'].append)
    worker.cancelled.connect(
        lambda: record.__setitem__('cancelled', record['cancelled'] + 1))
    return record


# ============================================================================
# CancellationToken
# ============================================================================

class TestCancellationToken:

    def test_starts_uncancelled(self):
        assert CancellationToken().is_cancelled() is False

    def test_cancel_sets_flag(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled() is True

    def test_reset(self):
        token = CancellationToken()
        token.cancel()
        token.reset()
        assert token.is_cancelled() is False

    def test_is_not_a_qobject(self):
        """Thread affinity is exactly what this type exists to avoid.

        A QObject's slot invoked from another thread is queued onto the
        worker's event loop, which is blocked during a render, so the
        cancel would never arrive in time.
        """
        from PyQt6.QtCore import QObject
        assert not isinstance(CancellationToken(), QObject)


# ============================================================================
# MosaicJob
# ============================================================================

class TestMosaicJob:

    def test_preprocessor_config_uses_tile_dimensions(self, job):
        config = job.preprocessor_config()
        assert config.target_width == job.tile_width
        assert config.target_height == job.tile_height

    def test_render_config_uses_tile_dimensions(self, job):
        config = job.render_config()
        assert config.tile_width == job.tile_width
        assert config.tile_height == job.tile_height

    def test_configs_agree_on_aspect_ratio(self, temp_dir, tile_folder, guide_file):
        """A mismatch here would distort every tile in the mosaic."""
        job = MosaicJob(guide_file, tile_folder,
                        os.path.join(temp_dir, 'o.png'),
                        tile_width=120, tile_height=80)
        assert (job.preprocessor_config().target_aspect_ratio ==
                pytest.approx(job.render_config().tile_aspect_ratio))

    def test_repetition_settings_reach_render_config(self, job):
        job.max_tile_reuse = 7
        job.min_reuse_distance = 3
        config = job.render_config()
        assert config.max_tile_reuse == 7
        assert config.min_reuse_distance == 3


# ============================================================================
# Stage weights
# ============================================================================

class TestStageWeights:

    def test_weights_sum_to_one(self):
        assert sum(STAGE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_order_matches_weights(self):
        assert STAGE_ORDER == list(STAGE_WEIGHTS)

    def test_tile_loading_is_the_dominant_stage(self):
        """Analysing a tile library is by far the slowest step; if it were
        not weighted heaviest the bar would stall then jump."""
        assert max(STAGE_WEIGHTS, key=STAGE_WEIGHTS.get) == "Loading tiles..."


# ============================================================================
# A successful run
# ============================================================================

class TestSuccessfulRun:

    def test_emits_finished_once(self, qapp, job):
        worker = MosaicWorker(job)
        record = collect(worker)
        worker.run()

        assert len(record['finished']) == 1
        assert record['failed'] == []
        assert record['cancelled'] == 0

    def test_writes_the_output_file(self, qapp, job):
        MosaicWorker(job).run()
        assert os.path.exists(job.output_path)

    def test_creates_missing_output_directory(self, qapp, job):
        assert not os.path.isdir(os.path.dirname(job.output_path))
        MosaicWorker(job).run()
        assert os.path.isdir(os.path.dirname(job.output_path))

    def test_output_matches_the_tile_grid(self, qapp, job):
        MosaicWorker(job).run()
        with Image.open(job.output_path) as img:
            assert img.size == (60, 40)

    def test_output_records_dpi(self, qapp, job):
        MosaicWorker(job).run()
        with Image.open(job.output_path) as img:
            dpi = img.info.get('dpi')
        assert dpi is not None
        # PNG stores pixels-per-metre, so the value round-trips approximately.
        assert round(dpi[0]) == job.output_dpi

    def test_reports_every_stage_in_order(self, qapp, job):
        worker = MosaicWorker(job)
        record = collect(worker)
        worker.run()

        seen = []
        for stage, _, _ in record['stages']:
            if not seen or seen[-1] != stage:
                seen.append(stage)
        assert seen == STAGE_ORDER

    def test_overall_progress_is_monotonic_and_complete(self, qapp, job):
        worker = MosaicWorker(job)
        record = collect(worker)
        worker.run()

        overall = record['overall']
        assert overall, "no overall progress was reported"
        assert all(a <= b for a, b in zip(overall, overall[1:]))
        assert 0 <= min(overall) and max(overall) == 100

    def test_stats_describe_the_plan(self, qapp, job):
        worker = MosaicWorker(job)
        record = collect(worker)
        worker.run()

        _path, stats = record['finished'][0]
        assert stats.total_cells == 6 * 4
        assert stats.distinct_tiles >= 1

    def test_saves_jpeg(self, qapp, job, temp_dir):
        job.output_path = os.path.join(temp_dir, 'mosaic.jpg')
        MosaicWorker(job).run()
        with Image.open(job.output_path) as img:
            assert img.format == 'JPEG'


# ============================================================================
# Failures
# ============================================================================

class TestFailures:

    def test_missing_tile_folder_reports_failure(self, qapp, job, temp_dir):
        job.tile_folder_path = os.path.join(temp_dir, 'absent')
        worker = MosaicWorker(job)
        record = collect(worker)
        worker.run()

        assert len(record['failed']) == 1
        assert record['finished'] == []

    def test_empty_tile_folder_reports_failure(self, qapp, job, temp_dir):
        empty = os.path.join(temp_dir, 'empty')
        os.makedirs(empty)
        job.tile_folder_path = empty

        worker = MosaicWorker(job)
        record = collect(worker)
        worker.run()

        assert len(record['failed']) == 1
        assert 'No usable tile images' in record['failed'][0]

    def test_missing_guide_image_reports_failure(self, qapp, job, temp_dir):
        job.guide_image_path = os.path.join(temp_dir, 'gone.png')
        worker = MosaicWorker(job)
        record = collect(worker)
        worker.run()

        assert len(record['failed']) == 1
        assert record['finished'] == []

    def test_failure_does_not_also_emit_finished(self, qapp, job, temp_dir):
        job.tile_folder_path = os.path.join(temp_dir, 'absent')
        worker = MosaicWorker(job)
        record = collect(worker)
        worker.run()

        assert record['finished'] == []
        assert record['cancelled'] == 0


# ============================================================================
# Cancellation
# ============================================================================

class TestCancellation:

    def test_cancelling_before_start_emits_cancelled(self, qapp, job):
        worker = MosaicWorker(job)
        record = collect(worker)
        worker.cancel()
        worker.run()

        assert record['cancelled'] == 1
        assert record['finished'] == []

    def test_cancelled_run_writes_no_output(self, qapp, job):
        worker = MosaicWorker(job)
        worker.cancel()
        worker.run()
        assert not os.path.exists(job.output_path)

    def test_shared_token_cancels_the_worker(self, qapp, job):
        token = CancellationToken()
        worker = MosaicWorker(job, token=token)
        record = collect(worker)

        token.cancel()
        worker.run()

        assert record['cancelled'] == 1

    def test_cancel_on_a_real_thread(self, qapp, job, temp_dir):
        """The threaded path: cancel must land while run() is executing."""
        # Enough tiles that loading is still in progress when cancel fires.
        folder = os.path.join(temp_dir, 'many')
        os.makedirs(folder)
        for i in range(300):
            Image.new('RGB', (200, 200), (i % 256, 0, 0)).save(
                os.path.join(folder, f't{i:03d}.png'))
        job.tile_folder_path = folder

        worker = MosaicWorker(job)
        thread = QThread()
        worker.moveToThread(thread)

        outcome = []
        worker.finished.connect(lambda p, s: outcome.append('finished'))
        worker.cancelled.connect(lambda: outcome.append('cancelled'))
        worker.failed.connect(lambda m: outcome.append('failed:' + m))
        for signal in (worker.finished, worker.cancelled, worker.failed):
            signal.connect(thread.quit)

        thread.started.connect(worker.run)
        thread.start()

        # The token has no thread affinity, so this takes effect at once
        # even though the worker thread is busy inside run().
        QTimer.singleShot(150, worker.token.cancel)
        QTimer.singleShot(60000, thread.quit)

        while thread.isRunning():
            qapp.processEvents()
        thread.wait()

        assert outcome == ['cancelled'], outcome
        assert not os.path.exists(job.output_path)
