"""The control trace a rollout writes into its Rerun recording, one row per tick."""
import logging
from queue import Full, Queue
import threading
import time

import rerun as rr

logger = logging.getLogger(__name__)
MAX_PENDING_EVENTS = 256

# Where the time goes between a worker response and the robot executing it, per accepted prediction.
# All are milliseconds. "Commanded" is the first row of the new plan sent to the robot (in a shadow run,
# the first row taken from the plan).
LATENCY_STAGES = (
    "prepare_ms",                 # request preparation; forward kinematics of state and committed rows for pose workers
    "predict_ms",                 # the worker call (request, inference, response)
    "ik_ms",                      # inverse kinematics of the new rows (pose workers only)
    "postprocess_ms",             # response to checked joint chunk: validation, IK, gripper clipping
    "ready_to_accept_ms",         # chunk waiting for the control loop; includes the prefetch buffer under splice
    "accept_to_command_ms",       # chunk accepted into the plan to its first row sent
    "response_to_command_ms",     # worker response to the first new row sent
    "observation_to_command_ms",  # the control loop reading the observation to the first new row sent
)


class RolloutTrace:
    """Queue samples on the camera timeline and serialize them on a writer thread.

    Callers pass arrays that remain unchanged after submission. A slow recorder
    can drop samples, and shutdown drains accepted samples before disconnect.
    """

    def __init__(self, recording, start_time, session, *, active_arm_side="left"):
        self.recording = recording
        self.start_time = start_time
        self._submit_lock = threading.Lock()
        self._pending = Queue(maxsize=MAX_PENDING_EVENTS)
        self._stopped = False
        self._dropped = 0
        self._plan_path = f"/robot/planned_{active_arm_side}"
        names = [f"{arm} joint {joint}" if joint < 7 else f"{arm} aperture"
                 for arm in ("left", "right") for joint in range(1, 8)]
        for path in ("measured", "command_sent"):
            recording.log(f"/robot/{path}", rr.SeriesLines(names=names), static=True)
        arm_names = names[:7] if active_arm_side == "left" else names[7:]
        recording.log(self._plan_path, rr.SeriesLines(names=arm_names), static=True)
        encoder = getattr(session, "encoder_label", None) or getattr(session, "encoder", "unknown")
        recording.log("/policy/encoder", rr.TextDocument(str(encoder)), static=True)
        self._writer = threading.Thread(target=self._write, name="rollout-trace", daemon=True)
        self._writer.start()

    def _submit(self, write, *args):
        elapsed = time.monotonic() - self.start_time
        with self._submit_lock:
            if self._stopped:
                return
            try:
                self._pending.put_nowait((elapsed, write, args))
            except Full:
                self._dropped += 1

    def _write(self):
        while True:
            event = self._pending.get()
            if event is None:
                return
            elapsed, write, args = event
            try:
                self.recording.set_time("elapsed", duration=elapsed)
                write(*args)
            except Exception:
                logger.exception("Could not write rollout trace sample")

    def prediction(self, prediction_id, response, chunk, session, elapsed, *, prepare_chunk=None):
        sizes = getattr(session, "payload_sizes", None)
        request_bytes = sizes[-1].get("request_bytes") if sizes else None
        self._submit(self._write_prediction, prediction_id, getattr(response, "timings", None),
                     chunk, request_bytes, elapsed, prepare_chunk)

    def _write_prediction(self, prediction_id, timings, chunk, request_bytes, elapsed, prepare_chunk):
        self.recording.log("/policy/latency/client_ms", rr.Scalars(elapsed * 1000))
        for field, label in (("accel_exec_us", "model_ms"), ("net_wire_us", "network_ms"),
                             ("encode_us", "encode_ms")):
            duration = getattr(timings, field, None)
            if duration is not None:
                self.recording.log(f"/policy/latency/{label}", rr.Scalars(duration / 1000))
        if request_bytes is not None:
            self.recording.log("/policy/upload/request_bytes", rr.Scalars(request_bytes))
        if prepare_chunk is not None:
            chunk = prepare_chunk(chunk)
        if chunk.ndim == 2 and chunk.size:
            self.recording.log("/policy/raw_actions", rr.Tensor(chunk, dim_names=["row", "action"]),
                               rr.AnyValues(prediction_id=prediction_id))
        self._write_event(f"Prediction {prediction_id} received")

    def execution_latency(self, prediction_id, stages):
        """Log where the time went from a worker response to the robot executing its first row."""
        self._submit(self._write_execution_latency, prediction_id, dict(stages))

    def _write_execution_latency(self, prediction_id, stages):
        for name, milliseconds in stages.items():
            self.recording.log(f"/policy/latency/{name}", rr.Scalars(milliseconds))
        self._write_event(f"Prediction {prediction_id} first row commanded "
                          f"{stages['response_to_command_ms']:.0f} ms after the response")

    def event(self, message, *args):
        self._submit(self._write_event, message, *args)

    def _write_event(self, message, *args):
        self.recording.log("/policy/events", rr.TextLog(message % args if args else message))

    def tick(self, measured, row, command, prediction_id, row_index, *, holding, tick_ms=None):
        """``tick_ms`` is the control loop's work from the start of the tick to the row being sent."""
        self._submit(self._write_tick, measured, row, command, prediction_id, row_index, holding, tick_ms)

    def _write_tick(self, measured, row, command, prediction_id, row_index, holding, tick_ms):
        self.recording.log("/robot/measured", rr.Scalars(measured))
        self.recording.log("/policy/holding", rr.Scalars(int(holding)))
        if tick_ms is not None:
            self.recording.log("/policy/latency/tick_to_command_ms", rr.Scalars(tick_ms))
        if row is not None:
            self.recording.log(self._plan_path, rr.Scalars(row),
                               rr.AnyValues(prediction_id=prediction_id, row_index=row_index))
        if command is not None:
            self.recording.log("/robot/command_sent", rr.Scalars(command),
                               rr.AnyValues(prediction_id=prediction_id, row_index=row_index))

    def stop(self):
        with self._submit_lock:
            if self._stopped:
                return
            self._stopped = True
        self._pending.put(None)
        self._writer.join()
        if self._dropped:
            logger.warning("Dropped %d rollout trace samples because the writer was behind", self._dropped)
