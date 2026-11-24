"""The GR00T rollout conditions the Worker on a gripper inside the checkpoint range and keeps the executed rows."""

from types import SimpleNamespace

import numpy as np
import pytest

from committed_range import GR00T_COMMITTED_GRIPPER_RANGE, gr00t_conditioning_prefix


def test_open_and_closed_gripper_are_clamped_for_conditioning_only():
    rows = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.02],
                     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5]], dtype=np.float32)
    original = rows.copy()
    conditioned = gr00t_conditioning_prefix(rows)
    lo, hi = GR00T_COMMITTED_GRIPPER_RANGE
    assert conditioned[:, 6].tolist() == [float(np.float32(hi)), lo, 0.5]
    assert np.array_equal(conditioned[:, :6], rows[:, :6])
    assert np.array_equal(rows, original)


class _RangeCheckingWorker:
    """Rejects a committed gripper outside the checkpoint's range, as a Worker without the platform clamp does."""

    grant = SimpleNamespace(io_spec=SimpleNamespace(cameras={"top": (4, 4)}))

    def __init__(self):
        self.received = None

    def predict(self, cameras, state, instruction, episode_start, committed_actions=None, **_):
        self.received = committed_actions
        if committed_actions is not None and not (0.0 <= committed_actions[:, 6]).all() \
                or committed_actions is not None and not (committed_actions[:, 6] <= 0.96).all():
            raise PermissionError("Step rejected: GR00T committed actions cannot round-trip through the checkpoint normalization")
        actions = np.tile(np.linspace(0.0, 0.5, 7, dtype=np.float32), (16, 1))
        if committed_actions is not None:
            actions[:len(committed_actions)] = committed_actions
        return SimpleNamespace(actions=actions)


def test_rollout_commits_a_fully_open_gripper_without_the_worker_rejecting_it():
    from rollout_loop import checked_chunk

    worker = _RangeCheckingWorker()
    executed = np.tile(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], dtype=np.float32), (4, 1))
    chunk, _ = checked_chunk(worker, {}, np.zeros(7, np.float32), "go", episode_start=False,
                             committed_actions=executed)
    assert worker.received[:, 6].max() <= 0.96
    assert np.array_equal(chunk[:4], executed)
    assert chunk[:4, 6].tolist() == [1.0] * 4


def test_other_models_still_send_the_rows_exactly_as_committed():
    from rollout_loop import checked_chunk

    worker = _RangeCheckingWorker()
    committed = np.tile(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5], dtype=np.float32), (2, 1))
    checked_chunk(worker, {}, np.zeros(7, np.float32), "go", episode_start=False, model_label="Rho",
                  committed_actions=committed)
    assert np.array_equal(worker.received, committed)
