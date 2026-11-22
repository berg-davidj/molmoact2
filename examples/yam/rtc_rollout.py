"""Real-time chunking on the YAM rig, with Orchestral's ``RealTimeChunking``.

Real-time chunking (Black, Galliker, Levine 2025, Algorithm 1) requests the
next prediction while the arm is still executing the current chunk. The rows the
arm will run during that prediction are sent along as *committed actions*. The model returns a
chunk that starts with those rows. The SDK measures prediction delay and adjusts the
number of rows to commit. If a prediction is late, the arm holds the last committed target.

    chunking = RealTimeChunking(session, control_hz, s_min=7, d_init=2)
    while True:
        row = chunking.action()            # this tick's row; None: hold
        if row is not None:
            send(row)
        if chunking.should_replan():       # s_min rows of the chunk have run, none in flight
            chunking.observe(inputs=...)   # predicted in the background
        time.sleep(max(0.0, chunking.t_next - session.clock.now()))

It needs a deployment that supports committed actions, such as the pi0.5 red-cap checkpoint
(``npow/pi05-yam-red-cap-full-7500``, 15 x 7 actions at 15 Hz). The
rig, cameras, safety mask and rest-pose return are the launcher's own; only the control loop is new.

    PYTHONPATH=examples/yam python examples/yam/rtc_rollout.py \\
      --config-path examples/yam/configs/pi05_bimanual_physical.yaml \\
      --right-config-path examples/yam/configs/pi05_right_primary.yaml \\
      --orchestral-deployment dep_xxxxxxxxxxxx --active-arm-side left --seconds 60

Ctrl-C ends the rollout, and so does a failed prediction unless ``--max-prediction-failures``
allows some (a long run over a real network). Requires an Orchestral SDK (``orchestral-labs``) with
``orchestral.RealTimeChunking``.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch_yaml_eval_molmoact import (  # noqa: E402
    _build_env,
    resolve_bimanual_execution_mask,
    return_to_captured_rest_pose,
    validate_bimanual_model_arm_order,
)
from frame_capture_time import session_capture_time  # noqa: E402
from orchestral_session_bridge import OrchestralDirectHost, OrchestralSessionHost  # noqa: E402
from rtc_camera_frames import RtcCameraFrames, frame_times  # noqa: E402

logger = logging.getLogger("rtc_rollout")
ARM_DIM = 7
RIG_CAMERAS = {"top": "front_camera", "left": "left_camera", "right": "right_camera"}


class RolloutTelemetry:
    """Collect timing on the predictor thread and report after the motors close."""

    def __init__(self, control_hz: float):
        self.period_ms = 1000 / control_hz
        self.predictions = []
        self.hold_ticks = 0
        self.observations = 0
        self.without_capture_time = 0  # observations sent with no camera capture time
        self.failed_predictions = 0  # failed predictions the loop held through or stopped on
        self.camera_pair_fallbacks = 0

    def report(self):
        completed = 0
        failed = 0
        late_count = 0
        prefixes = Counter()
        metrics = {}
        for chunk, late in list(self.predictions):
            late_count += late
            if chunk.error is not None:
                failed += 1
                continue
            completed += 1
            prefix_rows = len(chunk.request.get("committed_actions", ()))
            prefixes[prefix_rows] += 1
            prediction_ms = 1000 * (chunk.t_completed - chunk.t_sent)
            times = {"prediction_ms": prediction_ms}
            if "observation_prep_ms" in chunk.meta:
                times["observation_prep_ms"] = chunk.meta["observation_prep_ms"]
                times["client_queue_ms"] = 1000 * (chunk.t_sent - chunk.meta["queued_at"])
                times["observation_to_predict_ms"] = times["observation_prep_ms"] + times["client_queue_ms"]
            for name in ("camera_pair_wait_ms", "camera_skew_ms", "camera_delivery_ms"):
                if name in chunk.meta:
                    times[name] = chunk.meta[name]
            if chunk.meta.get("captured_at") is not None:
                # Camera frame captured -> chunk back on the robot, both on the Session clock.
                times["observation_to_action_ms"] = 1000 * (chunk.t_completed - chunk.meta["captured_at"])
                times["frame_age_at_queue_ms"] = 1000 * (chunk.meta["queued_at"] - chunk.meta["captured_at"])
                times["capture_to_predict_ms"] = 1000 * (chunk.t_sent - chunk.meta["captured_at"])
            if (timing := chunk.timings) is not None:
                times.update(
                    client_encode_ms=timing.encode_us / 1000,
                    transport_estimate_ms=timing.net_wire_us / 1000,
                    worker_decode_ms=timing.decode_us / 1000,
                    worker_queue_ms=timing.queue_us / 1000,
                    inference_ms=timing.accel_exec_us / 1000,
                )
                accounted_us = (timing.encode_us + timing.net_wire_us + timing.decode_us
                                + timing.queue_us + timing.accel_exec_us)
                times["unattributed_ms"] = max(0.0, prediction_ms - accounted_us / 1000)
            for name, value in times.items():
                metrics.setdefault(name, []).append(value)
            if late or prediction_ms >= 2 * self.period_ms or any(
                times.get(name, 0) >= self.period_ms
                for name in ("observation_prep_ms", "client_queue_ms")
            ):
                logger.info(
                    "RTC timing observation=%d prefix_rows=%d late=%s %s",
                    chunk.seq, prefix_rows, late,
                    " ".join(f"{name}={value:.1f}" for name, value in times.items()),
                )

        logger.info(
            "RTC timing summary predictions=%d failed=%d prefix_rows_histogram=%s "
                    "late_handoffs=%d hold_ticks=%d tick_ms=%.1f observations=%d without_capture_time=%d "
                    "camera_pair_fallbacks=%d",
            completed, failed, dict(sorted(prefixes.items())), late_count, self.hold_ticks, self.period_ms,
                    self.observations, self.without_capture_time, self.camera_pair_fallbacks,
        )
        for name, values in metrics.items():
            median, p95, p99 = np.percentile(values, (50, 95, 99))
            logger.info("RTC timing %s n=%d p50=%.1f p95=%.1f p99=%.1f max=%.1f",
                        name, len(values), median, p95, p99, max(values))


class VerifiedCommittedSession:
    """Expose capability to RTC after a successful probe on a legacy Worker."""

    supports_committed_actions = True

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)


def make_chunking(session: Any, control_hz: float, *, s_min: int, d_init: int, rebase_joints: int,
                  telemetry: RolloutTelemetry | None = None):
    """``RealTimeChunking`` for this rig.

    The red-cap endpoint predicts joint *deltas* from the state it was queried with. Committed
    rows are sent in the new prediction's frame, so their joint deltas are re-expressed from the
    old query state to the new one (the gripper is absolute and stays as is).
    """
    from orchestral import RealTimeChunking

    class YamRealTimeChunking(RealTimeChunking):
        def _frozen_rows(self, s):
            # A prediction can arrive before the next tick. Once only zero-tick
            # samples remain, the SDK would otherwise submit an empty prefix.
            return max(1, super()._frozen_rows(s))

        def _prepare(self, observation):
            extra = super()._prepare(observation)
            if extra and rebase_joints:
                old = self.rows.chunk.meta["query_state"][:rebase_joints]
                new = observation.meta["query_state"][:rebase_joints]
                committed = np.array(extra["committed_actions"], dtype=np.float32, copy=True)
                committed[:, :rebase_joints] += old - new
                extra = {**extra, "committed_actions": committed}
            return extra

        def _deliver(self, chunk):
            late_before = self.late
            delivered = super()._deliver(chunk)
            if telemetry is not None:
                telemetry.predictions.append((chunk, self.late > late_before))
            return delivered

    return YamRealTimeChunking(session, control_hz, s_min=s_min, d_init=d_init)


def next_row(chunking, telemetry: RolloutTelemetry, max_prediction_failures: int):
    """This tick's row, or None to hold.

    A failed prediction drops the running chunk, so the arm holds its last target. Up to
    ``max_prediction_failures`` of them hold this tick and let planning restart; the next one
    is raised and ends the rollout.
    """
    from orchestral import PredictionFailed

    try:
        return chunking.action()
    except PredictionFailed as failed:
        telemetry.failed_predictions += 1
        if telemetry.failed_predictions > max_prediction_failures:
            raise
        logger.warning("Prediction failure %d of %d allowed, holding: %s",
                       telemetry.failed_predictions, max_prediction_failures, failed)
        return None


def run(env, host, mask, *, control_hz: float, instruction: str, s_min: int, d_init: int,
        seconds: float, telemetry: RolloutTelemetry, max_prediction_failures: int = 0,
        camera_frames: RtcCameraFrames | None = None) -> dict:
    """The control loop: one row per tick, one prediction in flight, until ``seconds`` or Ctrl-C."""
    endpoint_dim = host.identity["state_dim"]
    if endpoint_dim == ARM_DIM:
        endpoint = slice(0, ARM_DIM) if mask.active_arm_side == "left" else slice(ARM_DIM, 2 * ARM_DIM)
    else:
        endpoint = slice(0, endpoint_dim)
    red_cap = host.policy_adapter == "pi05_yam_red_cap"
    session = host.session
    prediction_cameras = {role: RIG_CAMERAS[role] for role in host.identity["camera_mapping"]}
    chunking = make_chunking(VerifiedCommittedSession(session), control_hz, s_min=s_min, d_init=d_init,
                             rebase_joints=6 if red_cap else 0, telemetry=telemetry)
    # An SDK that measures its control-loop latencies from the capture time takes it as
    # `t_captured`; an older one does not know the argument, and it stays in `meta` only.
    sdk_takes_capture_time = "t_captured" in inspect.signature(chunking.observations.put).parameters
    if not sdk_takes_capture_time:
        logger.warning("This orchestral SDK takes no t_captured: its observation-to-action and reaction "
                       "spans start at the next tick, so Grafana's capture-time latency panels stay empty. "
                       "Update the orchestral checkout.")
    sent = 0
    deadline = session.clock.now() + seconds
    while session.clock.now() < deadline:
        with chunking.actions.lock:
            row = next_row(chunking, telemetry, max_prediction_failures)
            if row is not None:
                query_state = chunking.rows.chunk.meta["query_state"]
        measured = mask.validate_state(env.get_robot_state()["joint_positions"], name="measured_state")
        if row is None:
            telemetry.hold_ticks += 1  # first chunk, late or failed prediction: keep the last target
        else:
            target = measured.copy()
            target[endpoint] = host.robot_actions(row, query_state)
            env.step_command_only(mask.command_target(target, measured), wait=False)
            sent += 1
        if chunking.should_replan():
            meta = {}
            frames = None
            if camera_frames is not None:
                wait_started = session.clock.now()
                # Reserve half a command tick for dispatching the observation and scheduling the next row.
                wait_budget = max(0.0, chunking.t_next - wait_started - 0.5 / control_hz)
                frames = camera_frames.take(wait_budget)
                meta["camera_pair_wait_ms"] = 1000 * (session.clock.now() - wait_started)
                telemetry.camera_pair_fallbacks += frames is None
            prep_started = session.clock.now()
            if frames is None:
                obs = env.get_obs()
                captured_at = session_capture_time(env, session.clock, camera_names=prediction_cameras.values())
                images = {role: obs[f"{camera}_rgb"] for role, camera in prediction_cameras.items()}
            else:
                images = {role: frame.image for role, frame in frames.items()}
                timestamps = frame_times(frames.values())
                captured_at = session.clock.now() - (time.monotonic() - min(timestamps))
                meta["camera_skew_ms"] = 1000 * (max(timestamps) - min(timestamps))
                if all(frame.captured_at is not None for frame in frames.values()):
                    meta["camera_delivery_ms"] = 1000 * max(
                        frame.ready_at - frame.captured_at for frame in frames.values()
                    )
                obs = env.get_robot_state()
            state = mask.validate_state(obs["joint_positions"], name="policy input state")[endpoint]
            inputs, query_state = host.predict_inputs(images, state, cameras_prepared=frames is not None)
            queued_at = session.clock.now()
            meta.update(query_state=query_state, observation_prep_ms=1000 * (queued_at - prep_started),
                        queued_at=queued_at)
            capture_time = {}
            telemetry.observations += 1
            if captured_at is not None:
                meta["captured_at"] = captured_at
                if sdk_takes_capture_time:
                    capture_time["t_captured"] = captured_at
            else:
                telemetry.without_capture_time += 1
                if telemetry.without_capture_time == 1:
                    logger.warning("An observation has no camera capture time (a camera reported none, or "
                                   "the wall clock stepped back): it is left out of the robot's latency report, "
                                   "and its SDK spans start at the next tick (orc.loop.measured_from=observation), "
                                   "outside Grafana's capture-time panels. The summary counts them.")
            chunking.observe(inputs=inputs, instruction=instruction, meta=meta, **capture_time)
        time.sleep(max(0.0, chunking.t_next - session.clock.now()))
    return {"rows_sent": sent, "hold_ticks": telemetry.hold_ticks,
            "failed_predictions": telemetry.failed_predictions, "late_chunks": chunking.late,
            "delays_ticks": list(chunking.delays), "rows_run_before_takeover": list(chunking.k)}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-path", required=True, help="primary (left-arm) rig config")
    parser.add_argument("--right-config-path", required=True, help="secondary (right-arm) rig config")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--orchestral-deployment", help="managed deployment id (dep_...)")
    target.add_argument("--orchestral-grant", help="ConnectGrant token file (chmod 600)")
    parser.add_argument("--active-arm-side", choices=["left", "right"], default=None,
                        help="arm the policy drives; the other holds (default: config)")
    parser.add_argument("--execution-mode", choices=["active_arm_hold", "shadow"], default=None,
                        help="shadow holds both arms (default: config)")
    parser.add_argument("--policy-adapter", choices=["identity", "pi05_yam_red_cap"], default=None,
                        help="default: config eval.server.policy_adapter")
    parser.add_argument("--instruction", default=None, help="default: config storage.language_instruction")
    parser.add_argument("--control-hz", type=float, default=None, help="default: config hz (15 for red-cap)")
    parser.add_argument("--camera-fps", type=int, help="RTC V4L2 capture rate (default: rig config)")
    parser.add_argument("--s-min", type=int, default=7, help="rows of a chunk to run before the next request")
    parser.add_argument("--d-init", type=int, default=2, help="prediction delay in ticks until one is measured")
    parser.add_argument("--seconds", type=float, default=60.0, help="rollout length")
    parser.add_argument("--max-prediction-failures", type=int, default=0,
                        help="failed predictions to hold through before one ends the rollout (default 0: "
                             "the first one ends it)")
    args = parser.parse_args(argv)
    if args.camera_fps is not None and args.camera_fps < 1:
        parser.error("--camera-fps must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    left_cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)
    right_cfg = OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
    validate_bimanual_model_arm_order(left_cfg, right_cfg)
    eval_cfg = left_cfg.get("eval") or {}
    bimanual_cfg = eval_cfg.get("bimanual") or {}
    mask = resolve_bimanual_execution_mask(
        bimanual=True,
        active_arm_side=args.active_arm_side or bimanual_cfg.get("active_arm_side"),
        execution_mode=args.execution_mode or bimanual_cfg.get("execution_mode"),
        both_arm_max_delta=bimanual_cfg.get("both_arm_max_delta"),
    )
    if mask.active_arm_side == "both":
        raise SystemExit("this example drives one arm; use --active-arm-side left or right")
    control_hz = float(args.control_hz or left_cfg.get("hz", 15))
    instruction = args.instruction or (left_cfg.get("storage") or {}).get("language_instruction") or ""
    adapter = args.policy_adapter or (eval_cfg.get("server") or {}).get("policy_adapter", "identity")

    try:
        from orchestral import PredictionFailed, RealTimeChunking  # noqa: F401
    except ImportError as exc:
        parser.exit(1, f"RTC requires an Orchestral SDK with RealTimeChunking and PredictionFailed: {exc}\n")

    # Verify the Session before opening the motors.
    if args.orchestral_grant:
        host = OrchestralDirectHost(grant=args.orchestral_grant, instruction=instruction,
                                    policy_adapter=adapter)
    else:
        host = OrchestralSessionHost(deployment_id=args.orchestral_deployment, instruction=instruction,
                                     policy_adapter=adapter)
    identity = host.open()
    telemetry = None
    try:
        if not host.supports_committed_actions():
            raise SystemExit("this deployment does not support committed actions; real-time chunking needs them")
        if identity["state_dim"] not in (ARM_DIM, 2 * ARM_DIM):
            raise SystemExit(f"unsupported endpoint state width {identity['state_dim']}")
        horizon = identity["action_horizon"]
        if identity["action_dim"] != identity["state_dim"]:
            raise SystemExit("the endpoint's action and state widths must match")
        if not 1 <= args.d_init <= args.s_min <= horizon - args.d_init:
            parser.error(f"need 1 <= d-init <= s-min <= {horizon} - d-init")
        logger.info("Deployment: %s, %d x %d actions at %.0f Hz; %s",
                    identity.get("deployment_id"), identity["action_horizon"], identity["action_dim"],
                    control_hz, mask.execution_summary)
        roles = identity["camera_mapping"].keys()
        camera_cfg = left_cfg["sensors"]["cameras"]
        camera_fps = {}
        camera_frames = None
        if (not (eval_cfg.get("camera_server") or {}).get("enabled")
                and all(str(cfg["device_id"]).startswith("/dev/") for cfg in camera_cfg.values())
                and all(camera_cfg[RIG_CAMERAS[role]].get("enabled", True) for role in roles)):
            for role in roles:
                cfg = camera_cfg[RIG_CAMERAS[role]]
                camera_fps[role] = int(args.camera_fps or cfg.get("rtc_fps") or eval_cfg.get("rtc_camera_fps")
                                       or cfg.get("fps", 5))
            # A frame from the slower camera has a faster-camera frame within half a fast period
            # when both streams deliver on time. Keep the wait budget tied to the slower stream.
            skew_limit = 0.5 / max(camera_fps.values()) if len(roles) == 2 else None
            camera_frames = RtcCameraFrames(roles, 1 / min(camera_fps.values()), skew_limit=skew_limit)

        def prepare_camera_frame(role, image, ready_at, captured_at):
            # Resize on the capture threads so dispatch only needs to attach current joint feedback.
            camera_frames.publish(role, host.prepare_camera(role, image), ready_at, captured_at)

        env, *_ = _build_env(
            SimpleNamespace(config_path=args.config_path, right_config_path=args.right_config_path,
                            control_hz=control_hz, max_steps=None, camera_fps=camera_fps),
            camera_roles=roles,
            timed_camera_frame_callback=prepare_camera_frame if camera_frames is not None else None,
        )
        try:
            rest = np.asarray(env.get_robot_state()["joint_positions"], dtype=np.float32).copy()
            input(f"Instruction: {instruction!r}. Clear the workspace, then press Enter to start "
                  "(Ctrl-C ends the rollout)... ")
            telemetry = RolloutTelemetry(control_hz)
            completed = True
            try:
                summary = run(env, host, mask, control_hz=control_hz, instruction=instruction,
                              s_min=args.s_min, d_init=args.d_init, seconds=args.seconds, telemetry=telemetry,
                              max_prediction_failures=args.max_prediction_failures, camera_frames=camera_frames)
                logger.info("Rollout summary: %s", summary)
            except KeyboardInterrupt:
                logger.info("Interrupted")
            except Exception:
                completed = False  # a fault holds and releases the arms; it never starts a new motion
                raise
            finally:
                if completed and eval_cfg.get("return_to_rest_on_exit", False):
                    return_to_captured_rest_pose(
                        env, rest, max_joint_step=float(eval_cfg.get("return_to_rest_max_joint_step", 0.01)))
        finally:
            env.close()
    finally:
        try:
            host.close()
        finally:
            if telemetry is not None:
                telemetry.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
