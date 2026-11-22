"""Regressions for joining successive action chunks into one command stream."""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from chunk_fusion import MODES, ChunkPlan


def ramp(start, rows=16, step=0.01, gripper=0.0):
    """A chunk whose six joints all move `step` per row from `start`."""
    chunk = np.empty((rows, 7), dtype=np.float32)
    chunk[:, :6] = (start + step * np.arange(rows))[:, None]
    chunk[:, 6] = gripper
    return chunk


def drain(plan):
    rows = []
    while plan:
        rows.append(plan.pop())
    return rows


def run_stream(mode, offset, *, blend_rows=4):
    """Command 8 rows of one chunk, then add a second that disagrees with it by `offset`."""
    plan = ChunkPlan(mode, blend_rows=blend_rows)
    plan.add(ramp(0.0), 0)
    sent = [plan.pop()[1] for _ in range(8)]
    plan.add(ramp(0.08 + offset, gripper=1.0), 8)
    return np.array(sent + [row for _, row in drain(plan)])


def test_splice_appends_the_new_chunk_after_the_queued_rows():
    plan = ChunkPlan("splice")
    assert plan.add(ramp(0.0), 0) == 0
    for _ in range(8):
        plan.pop()
    # Requested with 8 rows queued, and added once those rows have been commanded.
    request_slot = plan.slot
    for _ in range(8):
        plan.pop()
    assert plan.add(ramp(1.0), request_slot) == 8
    assert [index for index, _ in drain(plan)] == list(range(8, 16))


def test_rows_for_slots_already_commanded_are_discarded():
    plan = ChunkPlan("crossfade")
    plan.add(ramp(0.0), 0)
    for _ in range(3):
        plan.pop()
    assert plan.add(ramp(0.0), 0) == 3
    assert plan.add(ramp(0.0, rows=3), 0) is None


@pytest.mark.parametrize("mode", MODES)
def test_agreeing_chunks_give_the_same_stream_in_every_mode(mode):
    stream = run_stream(mode, offset=0.0)
    np.testing.assert_allclose(stream[:, 0], 0.01 * np.arange(len(stream)), atol=1e-5)


@pytest.mark.parametrize("mode", ["crossfade", "ensemble", "rail"])
def test_fusing_modes_shrink_the_step_onto_a_disagreeing_chunk(mode):
    offset = 0.3
    largest_step = np.abs(np.diff(run_stream(mode, offset)[:, 0])).max()
    assert largest_step < 0.6 * offset


@pytest.mark.parametrize("mode", ["crossfade", "rail"])
def test_ramped_modes_end_on_the_new_chunk(mode):
    stream = run_stream(mode, offset=0.3, blend_rows=4)
    expected = 0.38 + 0.01 * np.arange(16)
    # Slots 8..11 are the ramp; from slot 12 on the stream is the new chunk.
    np.testing.assert_allclose(stream[12:, 0], expected[4:], atol=1e-4)


def test_ensemble_averages_every_prediction_for_a_slot():
    stream = run_stream("ensemble", offset=0.3)
    # Two predictions cover slots 8..15, nearly equally weighted; one covers slots 16 on.
    np.testing.assert_allclose(stream[8:16, 0], 0.01 * np.arange(8, 16) + 0.15, atol=1e-3)
    np.testing.assert_allclose(stream[16:, 0], 0.38 + 0.01 * np.arange(8, 16), atol=1e-5)


def test_rail_replaces_a_noisy_chunk_with_its_cubic_fit():
    noisy = ramp(0.0)
    noisy[:, :6] += np.random.default_rng(0).normal(0, 0.02, (16, 6)).astype(np.float32)
    plan = ChunkPlan("rail")
    plan.add(noisy, 0)
    smooth = np.array([row for _, row in drain(plan)])
    assert np.abs(np.diff(smooth[:, :6], n=2, axis=0)).max() < 0.25 * np.abs(np.diff(noisy[:, :6], n=2, axis=0)).max()


def test_rail_bridge_keeps_velocity_continuous():
    stream = run_stream("rail", offset=0.3, blend_rows=6)[:, 0]
    acceleration = np.abs(np.diff(stream, n=2))
    # A hard splice would show the whole 0.3 rad offset as one acceleration spike.
    assert acceleration.max() < 0.1


@pytest.mark.parametrize("mode", ["crossfade", "ensemble", "rail"])
def test_gripper_follows_the_newest_chunk(mode):
    stream = run_stream(mode, offset=0.3)
    np.testing.assert_array_equal(stream[:8, 6], 0.0)
    np.testing.assert_array_equal(stream[8:, 6], 1.0)


class _Session:
    """Returns chunks anchored on the state it is given, like the YAM joint checkpoints."""

    grant = SimpleNamespace(io_spec=SimpleNamespace(cameras={"top": (4, 4)}))

    def predict(self, *, cameras, state, instruction, episode_start):
        chunk = np.tile(np.asarray(state, dtype=np.float32), (16, 1))
        chunk[:, :6] += 0.01 * np.arange(16)[:, None]
        return SimpleNamespace(actions=chunk)


class _Env:
    def __init__(self):
        self.position = np.zeros(14, dtype=np.float32)
        self.commands = []

    def get_obs(self):
        return {"joint_positions": self.position.copy(), "front_camera_rgb": np.zeros((4, 4, 3), np.uint8)}

    def get_robot_state(self):
        return {"joint_positions": self.position.copy()}

    def step_command_only(self, command, wait=False):
        self.position = np.asarray(command, dtype=np.float32).copy()
        self.commands.append(self.position[:7].copy())


class _Mask:
    def validate_state(self, state, name):
        return np.asarray(state, dtype=np.float32)

    def command_target(self, target, measured):
        return target


class _InlinePredictor:
    """Finishes each prediction before the loop's next tick, so a run is deterministic."""

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, function, *args, **kwargs):
        future = Future()
        future.set_result(function(*args, **kwargs))
        return future

    def shutdown(self, **kwargs):
        pass


@pytest.mark.parametrize("mode", MODES)
def test_rollout_loop_commands_a_continuous_stream(mode, monkeypatch):
    groot_rollout = pytest.importorskip("groot_rollout")
    monkeypatch.setattr(groot_rollout, "ThreadPoolExecutor", _InlinePredictor)
    clock = iter(np.arange(0, 1000, 0.001))
    monkeypatch.setattr(groot_rollout.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(groot_rollout.time, "sleep", lambda seconds: None)
    env = _Env()
    groot_rollout.run(env, _Session(), _Mask(), seconds=0.5, instruction="go", shadow=False,
                      camera_bindings={"top": "front_camera_rgb"}, history_offsets=(),
                      chunk_fusion=mode, lead_steps=4)
    commands = np.array(env.commands)
    assert len(commands) > 40
    steps = np.diff(commands[:, 0])
    assert steps.min() > -1e-4 and steps.max() < 0.03
