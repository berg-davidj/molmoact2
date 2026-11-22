from types import SimpleNamespace

import numpy as np
import pytest

orchestral = pytest.importorskip("orchestral")

from rtc_rollout import RolloutTelemetry, next_row  # noqa: E402

ROW = np.zeros(7, dtype=np.float32)


class Chunking:
    """`RealTimeChunking.action()`: a failed prediction raises once, then the loop holds and goes on."""

    def __init__(self, ticks):
        self._ticks = iter(ticks)

    def action(self):
        tick = next(self._ticks)
        if isinstance(tick, str):
            raise orchestral.PredictionFailed(SimpleNamespace(seq=3, error=ConnectionError("link down")))
        return tick


def test_first_failed_prediction_ends_the_rollout_by_default():
    telemetry = RolloutTelemetry(control_hz=15)
    with pytest.raises(orchestral.PredictionFailed):
        next_row(Chunking(["failed"]), telemetry, 0)
    assert telemetry.failed_predictions == 1


def test_allowed_failures_hold_then_the_next_one_ends_the_rollout():
    telemetry = RolloutTelemetry(control_hz=15)
    chunking = Chunking([ROW, "failed", None, ROW, "failed", "failed"])
    rows = [next_row(chunking, telemetry, 2) for _ in range(5)]
    assert [row is None for row in rows] == [False, True, True, False, True]
    assert telemetry.failed_predictions == 2
    with pytest.raises(orchestral.PredictionFailed):
        next_row(chunking, telemetry, 2)
    assert telemetry.failed_predictions == 3
