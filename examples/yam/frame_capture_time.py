"""When the camera frames of an observation were captured, on the Session clock.

The camera drivers stamp each frame as they store it, on two clocks: ``time.time()`` and
``time.monotonic()``. ``RobotEnv.get_obs()`` keeps both for every frame it returned, in
``last_frame_timestamps`` and ``last_frame_monotonic`` (``None`` for a camera that reported none).
The Orchestral Session clock reads ``time.monotonic()`` by default, so the monotonic stamp maps
onto it directly and is used when it can be. Otherwise the wall-clock stamp is carried over as
an age, taken off the Session's current time: call this right after ``get_obs()``, so the two
clocks are read at the same moment.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Optional

_BOOT_ID = "/proc/sys/kernel/random/boot_id"


def boot_id() -> Optional[str]:
    """This machine's boot ID (Linux). Two processes with the same one share ``time.monotonic()``."""
    try:
        with open(_BOOT_ID) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _session_clock_is_monotonic(clock: Any) -> bool:
    """The Session clock reads ``time.monotonic()``: its source time now is the monotonic time now."""
    return abs(clock.to_source(clock.now()) - time.monotonic()) < 0.01


def session_capture_time(env: Any, clock: Any, *, camera_names: Iterable[str] | None = None) -> Optional[float]:
    """Session capture time of the oldest selected frame from the last ``env.get_obs()``.

    Pass the cameras used by the prediction to measure the age of its oldest input frame.
    With no selection, include all returned frames. Return None when a selected camera has
    no capture time or the oldest wall-clock stamp is ahead of the wall clock.
    """
    timestamps = getattr(env, "last_frame_timestamps", None) or {}
    names = tuple(timestamps if camera_names is None else camera_names)
    if not names or any(timestamps.get(name) is None for name in names):
        return None
    monotonic = getattr(env, "last_frame_monotonic", None) or {}
    if (all(monotonic.get(name) is not None for name in names)
            and _session_clock_is_monotonic(clock)):
        return clock.from_source(min(monotonic[name] for name in names))
    age = time.time() - min(timestamps[name] for name in names)
    if age < 0:
        return None
    return clock.now() - age
