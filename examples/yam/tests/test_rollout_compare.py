"""Ranking rollout settings from scored runs: the measures, the bookkeeping and the verdict."""
import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import rollout_compare
from rollout_compare import MIN_RUNS, main, motion_measures, read_sidecar, record_settings, summarize

HZ = 15.0


def ramp(ticks, speed=0.3, joints=7):
    """A plan moving every joint at `speed` rad/s, and the times of its ticks."""
    times = np.arange(ticks) / HZ
    return times, np.repeat((speed * times)[:, None], joints, axis=1)


def following(plan):
    """The arm reaching each command one tick after it was sent, already moving at the start."""
    return np.vstack([2 * plan[:1] - plan[1:2], plan[:-1]])


def test_an_arm_that_follows_a_steady_plan_scores_clean():
    times, plan = ramp(60)
    arm = following(plan)
    measures = motion_measures(times, arm, plan, plan, np.repeat(np.arange(6), 10), np.zeros(60))
    assert measures["tick_hz"] == pytest.approx(HZ)
    assert measures["tracking"] == pytest.approx(0, abs=1e-9)
    assert measures["shake"] == pytest.approx(0, abs=1e-6)
    assert measures["seam"] == pytest.approx(1.0)
    assert measures["capped"] == 0 and measures["hold"] == 0


def test_a_jump_at_each_chunk_change_shows_as_a_seam_and_as_shake():
    times, plan = ramp(60)
    prediction = np.repeat(np.arange(6), 10)
    jumpy = plan + 0.06 * prediction[:, None]  # 0.06 rad extra at every chunk change
    arm = following(jumpy)
    measures = motion_measures(times, arm, jumpy, jumpy, prediction, np.zeros(60))
    within = 0.3  # rad/s
    assert measures["seam"] == pytest.approx((within + 0.06 * HZ) / within)
    smooth = motion_measures(times, following(plan), plan, plan, prediction, np.zeros(60))
    assert measures["shake"] > 1.0 > smooth["shake"]


def test_units_are_per_second_so_control_rates_compare():
    measures = []
    for hz in (15.0, 30.0):
        times = np.arange(int(4 * hz)) / hz
        position = 0.5 * 0.8 * times ** 2  # 0.8 rad/s^2 on every joint
        arm = np.repeat(position[:, None], 7, axis=1)
        measures.append(motion_measures(times, arm, arm, arm, np.zeros(len(times)), np.zeros(len(times))))
    assert measures[0]["shake"] == pytest.approx(0.8, rel=1e-3)
    assert measures[1]["shake"] == pytest.approx(0.8, rel=1e-3)


def test_a_hold_is_counted_but_is_not_motion():
    times, plan = ramp(40)
    times = np.concatenate([times[:20], times[20:] + 1.0])  # the arm held for a second
    holding = np.concatenate([np.zeros(40), np.ones(15)])
    arm = following(plan)
    measures = motion_measures(times, arm, plan, plan, np.zeros(40), holding)
    assert measures["hold"] == pytest.approx(15 / 55)
    assert measures["shake"] == pytest.approx(0, abs=1e-6)


def test_a_capped_command_and_a_shadow_run():
    times, plan = ramp(40)
    sent = plan.copy()
    sent[10:20, 2] -= 0.01
    arm = following(plan)
    assert motion_measures(times, arm, plan, sent, np.zeros(40), [])["capped"] == pytest.approx(0.25)
    shadow = motion_measures(times, arm, plan, None, np.repeat(np.arange(4), 10), [])
    assert math.isnan(shadow["shake"]) and math.isnan(shadow["tracking"]) and shadow["seam"] == pytest.approx(1.0)
    # One chunk for the whole run: there is no seam to measure.
    assert math.isnan(motion_measures(times, arm, plan, plan, np.zeros(40), [])["seam"])


def run(success, shake=1.0, **settings):
    measures = {"shake": shake, "tracking": 0.01, "seam": 1.0, "capped": 0.0, "hold": 0.0}
    return {"settings": {"deployment": "d", "seconds": 300, **settings}, "shadow": False,
            "measures": measures, "success": success, "outcome": {}}


def test_combinations_are_the_settings_that_vary_and_the_outcome_ranks_them():
    runs = ([run(1.0, chunk_fusion="rail", fusion_rows=4, rollout_seed=i) for i in range(3)]
            + [run(1 / 3, shake=0.2, chunk_fusion="splice", fusion_rows=4, rollout_seed=i) for i in range(3)])
    varying, combinations, line = summarize(runs)
    assert varying == ["chunk_fusion"]  # fusion_rows is the same everywhere; seeds never count
    assert [c["key"] for c in combinations] == [("rail",), ("splice",)]  # the smoother splice still loses
    assert combinations[0]["scored"] == 3 and combinations[0]["success"] == pytest.approx(1.0)
    assert line.startswith("Best so far: chunk_fusion=rail")


def test_no_verdict_from_too_few_or_too_similar_runs():
    few = [run(1.0, chunk_fusion="rail"), run(0.0, chunk_fusion="splice")]
    assert f"at least {MIN_RUNS} scored runs" in summarize(few)[2]
    close = ([run(score, chunk_fusion="rail") for score in (1.0, 1 / 3, 2 / 3)]
             + [run(score, chunk_fusion="splice") for score in (2 / 3, 1 / 3, 2 / 3)])
    assert summarize(close)[2].startswith("Too close to call")
    unscored = [run(None, shake=2.0, chunk_fusion="rail"), run(None, shake=1.0, chunk_fusion="splice")]
    _, combinations, line = summarize(unscored)
    assert [c["key"] for c in combinations] == [("splice",), ("rail",)] and line.startswith("No verdict")
    assert summarize([run(1.0, chunk_fusion="rail")])[2].startswith("Only one combination")


@pytest.mark.parametrize("third_score", [None, 1 / 3])
def test_verdict_waits_for_a_third_combination_even_if_it_ranks_last(third_score):
    runs = ([run(1.0, chunk_fusion="rail") for _ in range(3)]
            + [run(2 / 3, chunk_fusion="splice") for _ in range(3)]
            + [run(third_score, chunk_fusion="ensemble")])
    _, _, line = summarize(runs)
    assert line.startswith("No verdict yet")
    assert f"each combination needs at least {MIN_RUNS}" in line


@pytest.mark.parametrize("lead", [None, 8])
def test_joint_rollout_records_resolved_prefetch_lead(lead, tmp_path, monkeypatch):
    import managed_rollout

    class SettingsSaved(Exception):
        pass

    def save_settings(path, settings):
        record_settings(path, settings)
        raise SettingsSaved

    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(managed_rollout, "record_settings", save_settings)
    recording = tmp_path / "run.rrd"
    args = SimpleNamespace(policy_family="gr00t", control_hz=None, prefetch_lead_steps=lead,
                           committed_actions=False, chunk_fusion="splice", fusion_rows=4, shadow=False,
                           instruction="go", rrd=recording, deployment="gr00t-test", seconds=30, noise_seed=None)
    identity = {"state_dim": 7, "action_horizon": 16, "action_dim": 7,
                "camera_mapping": {"top": "top", "left": "left", "right": "right"}}
    with pytest.raises(SettingsSaved):
        managed_rollout.rollout_joint_policy(None, identity, args)
    assert read_sidecar(recording)["settings"]["prefetch_lead_steps"] == 8


def test_settings_are_saved_beside_the_recording_and_scored_later(tmp_path, capsys):
    recording = tmp_path / "pi05-holesaw_20261122_150000.rrd"
    recording.write_bytes(b"")
    record_settings(recording, {"deployment": "pi05-holesaw", "chunk_fusion": "rail", "fusion_rows": 4,
                                "rrd": tmp_path})
    saved = read_sidecar(recording)
    assert saved["settings"]["chunk_fusion"] == "rail" and saved["outcome"] is None
    assert saved["settings"]["rrd"] == str(tmp_path)  # anything not plain JSON is kept as text
    main(["--dir", str(tmp_path), "score", "--placed", "2", "--note", "dropped the last one"])
    main(["--dir", str(tmp_path), "score", "150000", "--placed", "3"])  # a correction, by part of the name
    assert read_sidecar(recording)["outcome"] == {"placed": 3, "out_of": 3, "note": ""}
    with pytest.raises(SystemExit):
        main(["--dir", str(tmp_path), "score", "--placed", "4"])


def test_a_run_recorded_without_settings_can_be_given_them(tmp_path):
    recording = tmp_path / "gr00t-n17-holesaw-15hz-3cam_20261122_124153.rrd"
    recording.write_bytes(b"")
    assert read_sidecar(recording)["settings"] == {"deployment": "gr00t-n17-holesaw-15hz-3cam"}
    main(["--dir", str(tmp_path), "score", "--set", "chunk_fusion=splice", "--set", "fusion_rows=4"])
    assert read_sidecar(recording)["settings"] == {
        "deployment": "gr00t-n17-holesaw-15hz-3cam", "chunk_fusion": "splice", "fusion_rows": 4}


def test_saving_settings_never_stops_a_rollout(tmp_path, capsys):
    record_settings(tmp_path / "missing-directory" / "run.rrd", {"deployment": "d"})
    assert "Could not save" in capsys.readouterr().out


def test_recordings_written_by_the_rollout_trace_are_ranked(tmp_path, monkeypatch, capsys):
    rr = pytest.importorskip("rerun")
    import rollout_trace

    clock = [0.0]
    monkeypatch.setattr(rollout_trace.time, "monotonic", lambda: clock[0])
    scores = {"rail": (3, 3, 2), "splice": (1, 0, 1)}
    for mode, placed in scores.items():
        for repeat, score in enumerate(placed):
            path = tmp_path / f"pi05-test_2026112{repeat}_10000{len(mode)}.rrd"
            recording = rr.RecordingStream("test")
            recording.save(path)
            trace = rollout_trace.RolloutTrace(recording, 0.0, SimpleNamespace(encoder="h264"))
            times, plan = ramp(90)
            if mode == "splice":
                plan = plan + 0.05 * (np.arange(90) // 8)[:, None]
            arm = np.zeros(14)
            for tick, row in enumerate(plan):
                clock[0] = times[tick]
                if tick % 8 == 0:
                    chunk = plan[tick:tick + 8] - (0.0 if mode == "splice" else 0.01)
                    trace.event(f"Prediction {tick // 8} requested with 0 queued rows")
                    trace.prediction(tick // 8, None, chunk.astype(np.float32), SimpleNamespace(), 0.05)
                command = np.concatenate([row, np.zeros(7)])
                trace.tick(arm, row, command, tick // 8, tick % 8, holding=False)
                arm = command
            trace.stop()
            recording.disconnect()
            record_settings(path, {"deployment": "pi05-test", "chunk_fusion": mode, "fusion_rows": 4,
                                   "execution_mode": "active_arm_hold"})
            main(["--dir", str(tmp_path), "score", str(path), "--placed", str(score)])
    shadow = tmp_path / "pi05-test_20261129_100000.rrd"
    shadow.write_bytes(b"")
    record_settings(shadow, {"deployment": "pi05-test", "chunk_fusion": "ensemble", "execution_mode": "shadow"})
    unknown = tmp_path / "pi05-test_20261128_100000.rrd"  # recorded before settings were saved
    unknown.write_bytes(b"")
    main(["--dir", str(tmp_path), "score", str(unknown), "--placed", "3"])
    capsys.readouterr()
    main(["--dir", str(tmp_path), "--runs"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "pi05-test: 6 runs, 2 combinations"
    assert lines[1].split()[:4] == ["chunk_fusion", "runs", "scored", "placed"]
    rail, splice = lines[2].split(), lines[3].split()
    assert rail[:4] == ["rail", "3", "3", "89%"] and splice[:4] == ["splice", "3", "3", "22%"]
    columns = lines[1].split()
    rail, splice = dict(zip(columns, rail)), dict(zip(columns, splice))
    assert float(rail["seam"].rstrip("x")) == pytest.approx(1.0) and float(splice["seam"].rstrip("x")) > 2
    assert float(splice["shake"]) > float(rail["shake"])  # the seams shake the arm
    assert rail["latency"] == splice["latency"] == "50ms"
    assert rail["age"] == "229ms"  # rows 0..7 of each chunk run 0..7 ticks after its observation
    assert float(rail["shift"]) == pytest.approx(0.01) and float(splice["shift"]) == 0
    assert rail["grasps"] == "0.0" and rail["time"] == "-"
    assert "Best so far: chunk_fusion=rail" in out and "1 shadow runs" in out and "1 runs with no saved settings" in out
    assert json.loads(rollout_compare.sidecar_path(shadow).read_text())["settings"]["execution_mode"] == "shadow"


def test_trace_samples_do_not_wait_for_a_stalled_recording(monkeypatch, caplog):
    pytest.importorskip("rerun")
    import rollout_trace

    clock = [10.0]
    monkeypatch.setattr(rollout_trace, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(rollout_trace, "MAX_PENDING_EVENTS", 2)
    entered = threading.Event()
    release = threading.Event()
    timestamps = []
    paths = []
    converted_on = []

    def set_time(timeline, *, duration):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test recorder was never released")
        timestamps.append((timeline, duration))

    def log(path, *components, static=False):
        if not static:
            paths.append(path)

    def prepare_chunk(chunk):
        converted_on.append(threading.current_thread())
        return chunk

    recording = SimpleNamespace(set_time=set_time, log=log)
    trace = rollout_trace.RolloutTrace(recording, clock[0], SimpleNamespace())
    try:
        trace.event("first event")
        assert entered.wait(timeout=5)
        clock[0] += 1
        trace.tick(np.zeros(14), None, None, 0, 0, holding=True)
        trace.prediction(1, None, np.zeros((2, 7)), SimpleNamespace(), 0.05,
                         prepare_chunk=prepare_chunk)
        trace.event("overflow")
        assert not converted_on
        assert not paths
        clock[0] += 1
    finally:
        release.set()
        trace.stop()

    assert timestamps == [("elapsed", 0.0), ("elapsed", 1.0), ("elapsed", 1.0)]
    assert paths.count("/policy/events") == 2
    assert "/robot/measured" in paths
    assert "/policy/raw_actions" in paths
    assert len(converted_on) == 1 and converted_on[0] is not threading.current_thread()
    assert "Dropped 1 rollout trace samples" in caplog.text


def test_right_arm_trace_scores_the_right_arm(tmp_path, monkeypatch):
    rr = pytest.importorskip("rerun")
    import rollout_trace
    from rollout_trace_report import report

    clock = [0.0]
    monkeypatch.setattr(rollout_trace.time, "monotonic", lambda: clock[0])
    path = tmp_path / "right.rrd"
    recording = rr.RecordingStream("test")
    with recording:
        recording.save(path)
        trace = rollout_trace.RolloutTrace(recording, 0.0, SimpleNamespace(), active_arm_side="right")
        times, plan = ramp(40)
        for tick, (measured, target) in enumerate(zip(following(plan), plan)):
            clock[0] = times[tick]
            trace.tick(np.concatenate([np.zeros(7), measured]), target,
                       np.concatenate([np.zeros(7), target]), tick // 8, tick % 8, holding=False)
        trace.stop()
    recording.disconnect()

    measures = rollout_compare.measure_recording(path, shadow=False)
    assert measures["tracking"] == pytest.approx(0.0, abs=1e-6)
    assert measures["capped"] == 0.0
    assert "capped ticks       0%" in report(path)


def test_grasps_are_counted_whichever_way_the_gripper_closes():
    cycle = np.concatenate([np.full(10, 0.9), np.full(10, 0.1)])
    assert rollout_compare.count_grasps(np.tile(cycle, 3)) == 2  # closed three times, reopened twice
    assert rollout_compare.count_grasps(np.tile(1 - cycle, 3)) == 2
    assert rollout_compare.count_grasps(np.concatenate([np.tile(cycle, 3), [0.9]])) == 3
    assert rollout_compare.count_grasps(0.5 + 0.01 * np.random.default_rng(0).normal(size=200)) == 0
    # Dithering around the middle of the range is not a grasp.
    assert rollout_compare.count_grasps(np.concatenate([[0.0, 1.0], 0.5 + 0.05 * np.sin(np.arange(100))])) == 0


def test_age_shift_and_latency_come_from_the_predictions():
    times, plan = ramp(40)
    prediction = np.repeat(np.arange(1, 5), 10)
    row_index = np.tile(np.arange(5, 15), 4)  # each chunk's rows 5..14 ran
    arm = following(plan)
    # Real-time chunking: a row's index is the number of ticks since its observation.
    rtc = motion_measures(times, arm, plan, plan, prediction, [], row_index=row_index)
    assert rtc["age"] == pytest.approx(1000 * 9.5 / HZ)
    assert math.isnan(rtc["shift"]) and math.isnan(rtc["latency"])
    # The joint loop logs when each observation was taken: here 0.4 s before its first row ran.
    requested = {p: times[10 * (p - 1)] - 0.4 for p in range(1, 5)}
    raw = {}
    for p in range(1, 5):
        chunk = np.zeros((15, 7))
        chunk[5:15] = plan[10 * (p - 1):10 * p]
        chunk[5:15, :6] += 0.02  # the model's own rows sit 0.02 rad from what was commanded
        raw[p] = chunk
    joint = motion_measures(times, arm, plan, plan, prediction, [], row_index=row_index, requested=requested,
                            raw=raw, latency=[50, 60, 400])
    assert joint["age"] == pytest.approx(1000 * (0.4 + 4.5 / HZ))
    assert joint["shift"] == pytest.approx(0.02)
    assert joint["latency"] == 60
    # A raw chunk in another action space (a tool pose, both arms) cannot be compared with the plan.
    wide = motion_measures(times, arm, plan, plan, prediction, [], row_index=row_index,
                           raw={p: np.zeros((15, 14)) for p in range(1, 5)})
    assert math.isnan(wide["shift"])


def test_equal_outcomes_are_ordered_by_the_time_taken():
    fast = [run(1.0, chunk_fusion="rail") for _ in range(3)]
    slow = [run(1.0, shake=0.1, chunk_fusion="splice") for _ in range(3)]
    for member, seconds in zip(fast + slow, (40, 50, 45, 80, 90, 85)):
        member["outcome"] = {"time": seconds, "drops": 1}
    _, combinations, _ = summarize(slow + fast)
    assert [c["key"] for c in combinations] == [("rail",), ("splice",)]
    assert combinations[0]["time"] == pytest.approx(45) and combinations[0]["drops"] == pytest.approx(1)


def test_time_and_drops_are_part_of_a_score(tmp_path, capsys):
    recording = tmp_path / "pi05-holesaw_20261122_150000.rrd"
    recording.write_bytes(b"")
    with pytest.raises(SystemExit, match="--placed"):
        main(["--dir", str(tmp_path), "score", "--time", "48"])
    main(["--dir", str(tmp_path), "score", "--placed", "2", "--time", "48.5", "--drops", "1"])
    assert "2 of 3 placed in 48.5 s, 1 dropped" in capsys.readouterr().out
    main(["--dir", str(tmp_path), "score", "--drops", "0"])  # a correction keeps the rest
    assert read_sidecar(recording)["outcome"] == {"placed": 2, "out_of": 3, "note": "", "time": 48.5, "drops": 0}


@pytest.mark.parametrize("out_of", ["0", "-1"])
def test_score_rejects_nonpositive_out_of(tmp_path, out_of):
    recording = tmp_path / "run.rrd"
    recording.touch()
    with pytest.raises(SystemExit, match="--out-of must be positive"):
        main(["--dir", str(tmp_path), "score", "--placed", "0", "--out-of", out_of])
    assert not rollout_compare.sidecar_path(recording).exists()


def test_compare_skips_a_legacy_zero_denominator(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rollout_compare, "measure_recording", lambda *args, **kwargs: None)
    for name, out_of in (("bad", 0), ("good", 3)):
        recording = tmp_path / f"{name}.rrd"
        recording.touch()
        rollout_compare.sidecar_path(recording).write_text(json.dumps({
            "settings": {"deployment": "pi05-test", "chunk_fusion": "rail"},
            "outcome": {"placed": 0, "out_of": out_of},
        }))
    main(["--dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert "pi05-test: 1 runs, 1 combinations" in captured.out
    assert "bad.run.json: invalid run settings (out_of must be a positive integer)" in captured.err
