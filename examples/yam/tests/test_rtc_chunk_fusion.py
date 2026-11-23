"""Chunk fusion inside real-time chunking, over a real Session and the SDK's own RealTimeChunking.

The Worker runs in process with a small CPU model that honours committed actions, as the
Orchestral SDK's own real-time chunking tests do. The model answers in the red-cap frame: joint
rows are offsets from the state it was queried with. No robot or camera is opened.
"""
import threading

import numpy as np
import pytest

orchestral = pytest.importorskip("orchestral")

from chunk_fusion import MODES, ChunkPlan  # noqa: E402
from orchestral.core.contracts import IOSpec, PluginKind  # noqa: E402
from orchestral.core.registry import register  # noqa: E402
from rtc_rollout import RtcChunkFusion, main, make_chunking  # noqa: E402

H, WIDTH, CAMERA = 15, 7, "wrist"
LATE_DELAY = 6
FRAME = np.zeros((32, 32, 3), np.uint8)


def to_robot(actions, query_state):
    """The red-cap adapter's conversion: joint offsets from the query state, gripper inverted."""
    actions = np.array(actions, dtype=np.float32, copy=True)
    actions[..., :6] += query_state[:6]
    actions[..., 6] = np.clip(1.0 - actions[..., 6], 0.0, 1.0)
    return actions


def planned_targets(label):
    """Absolute joint targets of the chunk predicted from the state labelled `label`: a noisy ramp."""
    rows = np.arange(H, dtype=np.float32)[:, None]
    noise = np.random.default_rng(int(label)).normal(0, 0.02, (H, 6)).astype(np.float32)
    return 0.3 * label + 0.05 * rows + noise


@register(PluginKind.MODEL, "rtc-chunk-fusion-test")
class _RedCapModel:
    """state[0] labels the chunk and state[1] is the query offset; a negative label fails."""

    def load(self, checkpoint, device, config):
        return self

    def execution_evidence(self):
        return {"schema_version": 1, "execution_kind": "real", "regions": [{
            "name": "fixture", "runtime": "numpy", "device_kind": "cpu", "precision": "float32", "mechanisms": []}]}

    def io_spec(self):
        return IOSpec(cameras={CAMERA: (32, 32)}, state_dim=7, action_shape=(H, WIDTH))

    def forward(self, observation):
        state = np.asarray(observation["state"], np.float32)
        if state[0] < 0:
            raise ValueError("this observation fails")
        actions = np.full((H, WIDTH), 0.25, np.float32)
        actions[:, :6] = planned_targets(state[0]) - state[1]
        if state[0] == 123:
            actions[-1, 0] = np.nan
        return actions

    @property
    def supports_committed_actions(self):
        return True

    def forward_committed(self, observation, *, noise_seed=None):
        actions = self.forward(observation)
        prefix = observation["_orc_committed_actions"]
        actions[:len(prefix)] = prefix
        return actions


@pytest.fixture(scope="module")
def grant():
    from orchestral.runtime.serve import start_worker

    worker = start_worker(model_name="rtc-chunk-fusion-test", checkpoint="fixture://rtc-chunk-fusion",
                          accel_name="cpu", profile_name="eager", host="127.0.0.1", http_port=0, quic_port=0)
    yield worker.grant
    worker.close()


def rollout(grant, mode, *, late=None, fail=None, malformed=None, record_prediction=None,
            delay=2, ticks=60, fusion_rows=4):
    """The rig's chunking on a clock the test advances one tick at a time.

    A prediction requested in tick T is delivered just after tick T + delay; the one numbered
    `late` takes LATE_DELAY ticks, longer than the rows committed for it, so the arm holds. Returns the absolute
    row sent each tick (None: hold), every takeover as the fusion saw it, and the chunking.
    """
    from orchestral import PredictionFailed
    from orchestral.client.sdk import Session

    prepared, delivered = threading.Event(), threading.Event()
    takeovers = []

    class Recorded(RtcChunkFusion):
        def fuse(self, chunk, rows_run, *, after_hold):
            queued = [(index, row.copy()) for index, row in self._plan._queue]  # with each row's index
            fused = super().fuse(chunk, rows_run, after_hold=after_hold)
            query = chunk.meta["query_state"]
            takeovers.append(dict(rows_run=rows_run, after_hold=after_hold, queued=queued,
                                  raw=to_robot(chunk.actions, query), fused=to_robot(fused.actions, query)))
            return fused

    fusion = None if mode == "splice" else Recorded(mode, fusion_rows, to_robot)
    clock = [0.1]
    sent, requested, due, errors = [], 0, None, 0
    with Session(grant, clock=lambda: clock[0]) as session:
        chunking = make_chunking(session, 20.0, s_min=7, d_init=2, rebase_joints=6,
                                 fusion=fusion, record_prediction=record_prediction)
        prepare, deliver = chunking._prepare, chunking._deliver
        chunking.observations._prepare = lambda observation: (prepare(observation), prepared.set())[0]
        chunking.actions._on_deliver = lambda chunk: (deliver(chunk), delivered.set())[0]
        lock = chunking.actions.lock
        lock.acquire()  # deliveries wait for the test
        try:
            for tick in range(ticks):
                clock[0] = 0.1 + tick * 0.05
                try:
                    row = chunking.action()
                except PredictionFailed:
                    errors += 1
                    row = chunking.action()
                if row is None:
                    sent.append(None)
                else:
                    chunk = chunking.rows.chunk
                    sent.append((chunk.seq, chunking.rows.tick - chunk.tick,
                                 to_robot(row, chunk.meta["query_state"])))
                    if fusion is not None:
                        fusion.sent()
                if chunking.should_replan():
                    prepared.clear()
                    label = -1.0 if requested == fail else float(requested)
                    if requested == malformed:
                        label = 123.0
                    # Each query is taken from a different arm position, as on the rig.
                    query_state = np.full(7, 0.11 * requested, np.float32)
                    state = np.array([label, query_state[0], 0, 0, 0, 0, 0], np.float32)
                    chunking.observe(cameras={CAMERA: FRAME}, state=state, meta={"query_state": query_state})
                    assert prepared.wait(5)
                    requested, due = requested + 1, tick + (LATE_DELAY if requested == late else delay)
                if due == tick:
                    delivered.clear()
                    lock.release()
                    assert delivered.wait(5)
                    lock.acquire()
                    due = None
        finally:
            lock.release()
    return sent, takeovers, chunking, errors


def joints(sent):
    return [None if entry is None else entry[2][:6] for entry in sent]


def test_splice_is_real_time_chunking_unchanged(grant):
    sent, takeovers, chunking, _ = rollout(grant, "splice")
    assert not takeovers
    rows = [entry for entry in sent if entry is not None]
    assert len(rows) > 40 and chunking.late == 0
    # Every row is the model's own row for its chunk, apart from the committed rows it repeats.
    first = next(i for i, entry in enumerate(sent) if entry is not None)
    np.testing.assert_allclose(np.stack(joints(sent)[first:first + 7]), planned_targets(0)[:7], atol=1e-5)


@pytest.mark.parametrize("mode", [mode for mode in MODES if mode != "splice"])
@pytest.mark.parametrize(("late", "fail"), [(None, None), (2, None), (None, 2)])
def test_the_arm_runs_the_fused_rows_and_commits_them(grant, mode, late, fail):
    sent, takeovers, chunking, errors = rollout(grant, mode, late=late, fail=fail)
    assert errors == (fail is not None)
    assert len(takeovers) >= 4
    assert chunking.late == (late is not None)
    # Chunks arrive in order, so takeover n is the nth chunk the arm ran.
    by_seq = dict(zip(sorted({entry[0] for entry in sent if entry is not None}), takeovers))
    blended = 0
    for seq, index, row in (entry for entry in sent if entry is not None):
        takeover = by_seq[seq]
        # The row the SDK handed out is the fused plan's row for that slot.
        np.testing.assert_allclose(row[:6], takeover["fused"][index, :6], atol=1e-5)
        assert index >= takeover["rows_run"]
        # The gripper is never fused.
        np.testing.assert_allclose(row[6], takeover["raw"][index, 6], atol=1e-6)
    for previous, takeover in zip(takeovers, takeovers[1:]):
        k = takeover["rows_run"]
        if takeover["after_hold"] or not takeover["queued"]:
            continue
        # The plan the fusion kept is aligned with the SDK: its head is the old chunk's next row.
        head_index, head_row = takeover["queued"][0]
        np.testing.assert_allclose(head_row[:6], previous["fused"][head_index, :6], atol=1e-5)
        # The same fusion, recomputed from the two chunks alone.
        plan = ChunkPlan(mode, blend_rows=4)
        if mode == "crossfade":
            plan._replace([row for _, row in takeover["queued"]], head_index)
            plan.add(takeover["raw"], plan.slot - k)
            expected = plan.committed_actions()
            np.testing.assert_allclose(takeover["fused"][k:, :6], expected[:, :6], atol=1e-5)
        blended += not np.allclose(takeover["fused"][k:, :6], takeover["raw"][k:, :6], atol=1e-6)
    assert blended, "no takeover changed a row: the fusion did nothing"
    # Rows sent in consecutive ticks never jump further than the model's own row-to-row motion.
    steps = [np.abs(b - a).max() for a, b in zip(joints(sent), joints(sent)[1:]) if a is not None and b is not None]
    assert max(steps) < 0.25


def test_a_hold_or_failure_starts_a_new_plan(grant):
    for late, fail in ((2, None), (None, 2)):
        _, takeovers, _, _ = rollout(grant, "crossfade", late=late, fail=fail)
        restarted = [t for t in takeovers[1:] if t["after_hold"] or not t["queued"]]
        assert restarted
        for takeover in restarted:
            k = takeover["rows_run"]
            np.testing.assert_allclose(takeover["fused"][k:], takeover["raw"][k:], atol=1e-5)


@pytest.mark.parametrize("mode", ["splice", "rail"])
def test_nonfinite_prediction_is_reported_and_the_next_prediction_runs(grant, mode):
    recorded = []
    sent, _, _, errors = rollout(grant, mode, malformed=2,
                                 record_prediction=lambda chunk: recorded.append(chunk.seq))
    assert errors == 1
    assert 2 not in recorded
    assert any(row is not None and row[0] > 2 for row in sent)


def test_trace_exception_does_not_interrupt_action_delivery(grant, caplog):
    def record_prediction(chunk):
        if chunk.seq == 2:
            raise OSError("trace unavailable")

    sent, _, _, errors = rollout(grant, "splice", record_prediction=record_prediction)
    assert errors == 0
    assert "trace unavailable" in caplog.text
    assert {2, 3} <= {row[0] for row in sent if row is not None}


def test_fusion_needs_a_single_arm_endpoint():
    from types import SimpleNamespace

    from rtc_rollout import RolloutTelemetry, run

    host = SimpleNamespace(identity={"state_dim": 14, "action_dim": 14}, policy_adapter="identity")
    with pytest.raises(SystemExit, match="single-arm"):
        run(None, host, SimpleNamespace(active_arm_side="left"), control_hz=15, instruction="", s_min=7,
            d_init=2, seconds=1, telemetry=RolloutTelemetry(15), chunk_fusion="rail")


@pytest.mark.parametrize("arguments", [["--chunk-fusion", "bogus"], ["--fusion-rows", "0"], ["--chunk-fusion"]])
def test_invalid_fusion_arguments_are_refused_before_anything_opens(arguments, capsys):
    with pytest.raises(SystemExit) as refused:
        main(["--config-path", "missing.yaml", "--right-config-path", "missing.yaml",
              "--orchestral-deployment", "none", *arguments])
    assert refused.value.code == 2
    assert "fusion" in capsys.readouterr().err
