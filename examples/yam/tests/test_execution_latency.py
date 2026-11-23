"""The rollout measures where time goes from a worker response to the robot executing its first row."""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

PREDICT_SECONDS = 0.100
IK_SECONDS = 0.030
FK_SECONDS = 0.002


class _Clock:
    """Simulated time: only the fakes and the loop's sleep move it forward."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class _Session:
    grant = SimpleNamespace(io_spec=SimpleNamespace(cameras={"top": (4, 4)}))

    def __init__(self, clock):
        self.clock = clock

    def predict(self, *, state, **_kwargs):
        self.clock.now += PREDICT_SECONDS
        chunk = np.tile(np.asarray(state, dtype=np.float32), (16, 1))
        chunk[:, :6] += 0.01 * np.arange(16)[:, None]
        return SimpleNamespace(actions=chunk)


class _PoseActions:
    """Stands in for a tool-pose adapter; the identity map, with the cost of FK and IK."""

    def __init__(self, clock):
        self.clock = clock
        self.solved = 0

    def state(self, joints, *, matrix_policy=False):
        self.clock.now += FK_SECONDS
        return np.asarray(joints, dtype=np.float32)

    def actions(self, rows, start, *, matrix_policy=False):
        self.clock.now += IK_SECONDS
        self.solved += 1
        return np.asarray(rows, dtype=np.float32)


class _Env:
    def __init__(self):
        self.position = np.zeros(14, dtype=np.float32)
        self.commands = 0

    def get_obs(self):
        return {"joint_positions": self.position.copy(), "front_camera_rgb": np.zeros((4, 4, 3), np.uint8)}

    def get_robot_state(self):
        return {"joint_positions": self.position.copy()}

    def step_command_only(self, command, wait=False):
        self.position = np.asarray(command, dtype=np.float32).copy()
        self.commands += 1


class _Mask:
    def validate_state(self, state, name):
        return np.asarray(state, dtype=np.float32)

    def command_target(self, target, measured):
        return target


class _InlinePredictor:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, function, *args, **kwargs):
        future = Future()
        future.set_result(function(*args, **kwargs))
        return future

    def shutdown(self, **kwargs):
        pass


class _Trace:
    def __init__(self):
        self.latency = []
        self.tick_ms = []

    def prediction(self, *args, **kwargs):
        pass

    def event(self, *args):
        pass

    def tick(self, *args, holding, tick_ms=None):
        if tick_ms is not None:
            self.tick_ms.append(tick_ms)

    def execution_latency(self, prediction_id, stages):
        self.latency.append((prediction_id, stages))


@pytest.fixture
def clock(monkeypatch):
    groot_rollout = pytest.importorskip("groot_rollout")
    clock = _Clock()
    monkeypatch.setattr(groot_rollout, "ThreadPoolExecutor", _InlinePredictor)
    monkeypatch.setattr(groot_rollout.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(groot_rollout.time, "sleep", clock.sleep)
    return clock


def test_checked_chunk_times_each_stage_including_ik(clock):
    from groot_rollout import checked_chunk, execution_latency

    timings = {}
    _, elapsed = checked_chunk(_Session(clock), {}, np.zeros(7, np.float32), "go", episode_start=True,
                               pose_actions=_PoseActions(clock), timings=timings)
    clock.now += 0.010  # waiting for the control loop
    accepted_at = clock.now
    clock.now += 0.001
    stages = execution_latency(timings, observed_at=0.0, accepted_at=accepted_at, commanded_at=clock.now)
    assert elapsed == pytest.approx(FK_SECONDS + PREDICT_SECONDS + IK_SECONDS)
    assert stages["prepare_ms"] == pytest.approx(1000 * FK_SECONDS)
    assert stages["predict_ms"] == pytest.approx(1000 * PREDICT_SECONDS)
    assert stages["ik_ms"] == pytest.approx(1000 * IK_SECONDS)
    assert stages["postprocess_ms"] == pytest.approx(1000 * IK_SECONDS)
    assert stages["ready_to_accept_ms"] == pytest.approx(10)
    assert stages["accept_to_command_ms"] == pytest.approx(1)
    assert stages["response_to_command_ms"] == pytest.approx(1000 * IK_SECONDS + 11)
    assert stages["observation_to_command_ms"] == pytest.approx(1000 * (FK_SECONDS + PREDICT_SECONDS + IK_SECONDS) + 11)


def test_joint_worker_has_no_ik_stage(clock):
    from groot_rollout import checked_chunk

    timings = {}
    checked_chunk(_Session(clock), {}, np.zeros(7, np.float32), "go", episode_start=True, timings=timings)
    assert "ik_seconds" not in timings
    assert timings["ready"] - timings["responded"] == 0


@pytest.mark.parametrize("mode", ["splice", "crossfade"])
def test_rollout_traces_response_to_command_latency_per_accepted_prediction(clock, mode, capsys):
    import groot_rollout

    pose, trace, env = _PoseActions(clock), _Trace(), _Env()
    groot_rollout.run(env, _Session(clock), _Mask(), seconds=3, instruction="go", shadow=False,
                      camera_bindings={"top": "front_camera_rgb"}, history_offsets=(), lead_steps=4,
                      pose_actions=pose, trace=trace, chunk_fusion=mode, max_plan_rows=8)
    assert len(trace.latency) >= 3
    assert [prediction_id for prediction_id, _ in trace.latency] == list(range(1, len(trace.latency) + 1))
    for _, stages in trace.latency:
        assert stages["predict_ms"] == pytest.approx(1000 * PREDICT_SECONDS)
        assert stages["ik_ms"] == pytest.approx(1000 * IK_SECONDS)
        assert stages["response_to_command_ms"] == pytest.approx(
            stages["postprocess_ms"] + stages["ready_to_accept_ms"] + stages["accept_to_command_ms"])
        assert stages["observation_to_command_ms"] >= (
            stages["prepare_ms"] + stages["predict_ms"] + stages["response_to_command_ms"])
    assert len(trace.tick_ms) == env.commands
    summary = capsys.readouterr().out
    assert f"Latency in ms over {len(trace.latency)} accepted predictions" in summary
    for stage in ("ik_ms", "response_to_command_ms", "tick_to_command_ms"):
        assert stage in summary


class _SpanRecorder:
    def __init__(self):
        self.spans = []

    def __call__(self, name, start_ns, end_ns, *, attributes=None):
        self.spans.append((name, start_ns, end_ns, attributes))


def _traced_session(clock):
    session = _Session(clock)
    session._trace_attributes = lambda: {"orc.deployment_id": "dep-1", "orc.session_id": "s-1"}
    return session


def _run(groot_rollout, clock, session, **kwargs):
    trace = _Trace()
    groot_rollout.run(_Env(), session, _Mask(), seconds=3, instruction="go", shadow=False,
                      camera_bindings={"top": "front_camera_rgb"}, history_offsets=(), lead_steps=4,
                      pose_actions=_PoseActions(clock), trace=trace, max_plan_rows=8, model_label="rho", **kwargs)
    return trace


def test_each_accepted_prediction_is_a_grafana_span(clock, monkeypatch):
    import groot_rollout
    from orchestral.core import tracing

    recorder = _SpanRecorder()
    monkeypatch.setattr(tracing, "record_span", recorder)
    trace = _run(groot_rollout, clock, _traced_session(clock))
    assert len(recorder.spans) == len(trace.latency) >= 3
    for (name, start_ns, end_ns, attributes), (prediction_id, stages) in zip(recorder.spans, trace.latency):
        assert name == "yam.rollout.response_to_command"
        assert (end_ns - start_ns) / 1e6 == pytest.approx(stages["response_to_command_ms"], abs=1e-3)
        assert attributes["orc.deployment_id"] == "dep-1"
        assert attributes["yam.rollout.family"] == "rho"
        assert attributes["yam.rollout.prediction_id"] == prediction_id
        assert attributes["yam.rollout.ik_ms"] == pytest.approx(1000 * IK_SECONDS)
        assert attributes["yam.rollout.predict_ms"] == pytest.approx(1000 * PREDICT_SECONDS)


def test_a_failing_span_never_stops_the_rollout(clock, monkeypatch, capsys):
    import groot_rollout
    from orchestral.core import tracing

    calls = []

    def broken(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("exporter down")

    monkeypatch.setattr(tracing, "record_span", broken)
    trace = _run(groot_rollout, clock, _Session(clock))
    assert len(trace.latency) >= 3
    assert len(calls) == 1
    assert "Stopped recording latency spans: exporter down" in capsys.readouterr().out


def test_an_sdk_without_span_recording_keeps_the_rerun_trace(clock, monkeypatch, capsys):
    import groot_rollout
    from orchestral.core import tracing

    monkeypatch.delattr(tracing, "record_span")
    trace = _run(groot_rollout, clock, _Session(clock))
    assert len(trace.latency) >= 3
    assert "cannot record spans" in capsys.readouterr().out
