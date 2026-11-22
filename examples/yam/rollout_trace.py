"""The control trace a rollout writes into its Rerun recording, one row per tick."""
import threading
import time

import rerun as rr


class RolloutTrace:
    """Record policy responses and control ticks on the camera timeline."""

    def __init__(self, recording, start_time, session):
        self.recording = recording
        self.start_time = start_time
        self._response_lock = threading.Lock()
        self._stopped = False
        names = [f"{arm} joint {joint}" if joint < 7 else f"{arm} aperture"
                 for arm in ("left", "right") for joint in range(1, 8)]
        for path in ("measured", "command_sent"):
            recording.log(f"/robot/{path}", rr.SeriesLines(names=names), static=True)
        recording.log("/robot/planned_left", rr.SeriesLines(names=names[:7]), static=True)
        encoder = getattr(session, "encoder_label", None) or getattr(session, "encoder", "unknown")
        recording.log("/policy/encoder", rr.TextDocument(str(encoder)), static=True)

    def _set_time(self):
        self.recording.set_time("elapsed", duration=time.monotonic() - self.start_time)

    def prediction(self, prediction_id, response, chunk, session, elapsed):
        with self._response_lock:
            if self._stopped:
                return
            self._set_time()
            self.recording.log("/policy/latency/client_ms", rr.Scalars(elapsed * 1000))
            timings = getattr(response, "timings", None)
            for field, label in (("accel_exec_us", "model_ms"), ("net_wire_us", "network_ms"),
                                 ("encode_us", "encode_ms")):
                duration = getattr(timings, field, None)
                if duration is not None:
                    self.recording.log(f"/policy/latency/{label}", rr.Scalars(duration / 1000))
            sizes = getattr(session, "payload_sizes", None)
            if sizes and (request_bytes := sizes[-1].get("request_bytes")) is not None:
                self.recording.log("/policy/upload/request_bytes", rr.Scalars(request_bytes))
            if chunk.ndim == 2 and chunk.size:
                self.recording.log("/policy/raw_actions", rr.Tensor(chunk, dim_names=["row", "action"]),
                                   rr.AnyValues(prediction_id=prediction_id))
            self.recording.log("/policy/events", rr.TextLog(f"Prediction {prediction_id} received"))

    def event(self, message):
        self._set_time()
        self.recording.log("/policy/events", rr.TextLog(message))

    def tick(self, measured, row, command, prediction_id, row_index, *, holding):
        self._set_time()
        self.recording.log("/robot/measured", rr.Scalars(measured))
        self.recording.log("/policy/holding", rr.Scalars(int(holding)))
        if row is not None:
            self.recording.log("/robot/planned_left", rr.Scalars(row),
                               rr.AnyValues(prediction_id=prediction_id, row_index=row_index))
        if command is not None:
            self.recording.log("/robot/command_sent", rr.Scalars(command),
                               rr.AnyValues(prediction_id=prediction_id, row_index=row_index))

    def stop(self):
        with self._response_lock:
            self._stopped = True
