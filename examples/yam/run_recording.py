"""Run recordings for YAM rollouts, through the Orchestral SDK's ``RunRecorder``.

While a rollout runs inside ``recording(args, ...)``, the SDK records every Prediction (its times
and stages), chunk, tick and real-time-chunking decision by itself; the rollout adds the camera
frames, the commanded action of each tick, its own timed sections (``span``) and its chunk-plan
decisions. Render and explain a run afterwards with

    orchestral-operator bench recording explain yam_eval_runs/runs/<run_id>
    orchestral-operator bench recording render yam_eval_runs/runs/<run_id> --clock wall

An SDK without ``RunRecorder`` records nothing (one warning); the rollout runs the same.
"""
import contextlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
LEVELS = ("off", "summary", "timing", "full")
UPLOADS = ("off", "after", "live")


class _Off:
    """Stands in for a recorder when recording is off or the SDK has none: every call is a no-op."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def span(self, name, **fields):
        return contextlib.nullcontext()


OFF = _Off()


def add_arguments(parser):
    group = parser.add_argument_group("run recording (orchestral RunRecorder)")
    group.add_argument("--record", choices=LEVELS, default="full",
                       help="what to record: full (with camera images), timing (no images), summary, off")
    group.add_argument("--record-dir", type=Path, default=Path("yam_eval_runs/runs"),
                       help="folder of run recordings (one subfolder per run)")
    group.add_argument("--record-upload", choices=UPLOADS, default="after",
                       help="upload after the run, live during it, or keep it local (needs a robot API key)")
    group.add_argument("--record-max-hz", type=float, default=30.0,
                       help="camera images kept per camera per second (0: every frame)")
    return group


def current():
    """The SDK's current recorder, or a no-op."""
    try:
        from orchestral.client.recording import current as sdk_current
    except ImportError:
        return OFF
    return sdk_current()


def ns(seconds):
    """Monotonic seconds as integer nanoseconds (None stays None)."""
    return None if seconds is None else int(seconds * 1e9)


def predict_link(handle):
    """``Session.predict`` arguments that link a Prediction to its recorded observation."""
    return {} if handle is None else {"observation": handle}


@contextlib.contextmanager
def recording(args, **run_info):
    """Record the run (``args`` from ``add_arguments``); yields the recorder or a no-op."""
    if getattr(args, "record", "off") == "off":
        yield OFF
        return
    try:
        from orchestral import RunRecorder
    except ImportError:
        logger.warning("This orchestral SDK has no RunRecorder: the run is not recorded. Update the orchestral checkout.")
        yield OFF
        return
    upload = {"off": False}.get(args.record_upload, args.record_upload)
    with RunRecorder(args.record_dir, level=args.record, upload=upload, max_hz=args.record_max_hz) as recorder:
        recorder.run_info(**run_info)
        logger.info("Recording the run to %s", recorder.path)
        yield recorder
    print(f"[record] {recorder.path}\n"
          f"  orchestral-operator bench recording explain {recorder.path}\n"
          f"  orchestral-operator bench recording render {recorder.path} --clock wall", flush=True)
