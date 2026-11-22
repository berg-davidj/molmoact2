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


def session_capture_time(env: Any, clock: Any) -> Optional[float]:
    """Session time at which the oldest frame of the last ``env.get_obs()`` was captured.

    The oldest frame is used because an observation is only as fresh as its stalest camera.
    None, so no figure rather than a wrong one, when any camera reported no capture time (the
    oldest of the others would be too young), or when a wall-clock stamp is ahead of the wall
    clock (the clock was set back since the capture).
    """
    timestamps = getattr(env, "last_frame_timestamps", None)
    if not timestamps or any(ts is None for ts in timestamps.values()):
        return None
    monotonic = getattr(env, "last_frame_monotonic", None) or {}
    if (set(monotonic) == set(timestamps) and all(m is not None for m in monotonic.values())
            and _session_clock_is_monotonic(clock)):
        return clock.from_source(min(monotonic.values()))
    age = time.time() - min(timestamps.values())
    if age < 0:
        return None
    return clock.now() - age
